import argparse
import os
import torch
import datetime
import numpy as np
import pandas as pd
import lib.utils as utils
from src.OTFlowProblem import *
from src.Phi import *
from datasets.shallow_water import load_swdata

parser = argparse.ArgumentParser('COT-Flow')

parser.add_argument('--data', type=str, default='sw')
parser.add_argument("--nt_val", type=int, default=32, help="number of time steps for validation")
parser.add_argument('--nTh', type=int, default=2)
parser.add_argument('--dx', type=int, default=14, help="number of dimensions for x")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--prec', type=str, default='single', choices=['single', 'double'], help="single or double precision")
parser.add_argument('--num_trials', type=int, default=100, help="pilot run number of trials")
parser.add_argument('--save', type=str, default='experiments/cnf/tabcond')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# add timestamp to save path
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

if args.prec == 'double':
    prec = torch.float64
else:
    prec = torch.float32


def compute_loss(net, x, y, nt):
    Jc, cs = OTFlowProblem(x, y, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    columns_params = ["alpha1", "alpha2", "nt", "m", "m_y", "m_yout", "lr", "batchsz", "num_steps"]
    columns_valid = ["cx"]
    params_hist = pd.DataFrame(columns=columns_params)
    valid_hist = pd.DataFrame(columns=columns_valid)

    log_msg = ('{:5s}  {:9s}'.format('trial', ' valCx'))
    logger.info(log_msg)

    # box constraints / acceptable range for parameter values
    clampMax = 1.5
    clampMin = -1.5

    for trial in range(args.num_trials):

        alpha_NLL = 10 ** np.random.uniform(2, 5)
        alpha_HJB = 10 ** np.random.uniform(2, 5)
        alpha = [1.0, alpha_NLL, alpha_HJB]
        batch_size = 2 ** np.random.randint(7, 11)
        learning_rate = 10 ** np.random.randint(-4, -1)
        nt = 2 ** np.random.randint(3, 5)
        m = 2 ** np.random.randint(9, 11)
        my = 2 ** np.random.randint(5, 8)
        myout = 2 ** np.random.randint(5, 8)
        num_steps = 2 ** np.random.randint(3, 5)
        num_epochs = np.ceil(args.num_epochs / num_steps)
        # convert to int
        num_epochs = int(num_epochs)

        # load data
        train_loader, valid_loader, _, n_feat, _, _, Vy = load_swdata(batch_size, full=False)

        d = n_feat
        dx = args.dx
        if dx < 100:
            x_full = train_loader.dataset[:, :100]
            x_full = x_full.view(-1, 100)
            cov_x = x_full.T @ x_full
            L, V = torch.linalg.eigh(cov_x)
            # get the last dx columns in V
            Vx = cvt(V[:, -dx:])
        else:
            Vx = cvt(torch.eye(100))
        dy = d - 100

        params_hist.loc[len(params_hist.index)] = [alpha_NLL, alpha_HJB, nt, m, my, myout, learning_rate, batch_size, num_steps]

        nt_val = args.nt_val
        nTh = args.nTh

        # set up neural network to model potential function Phi
        net_y = nn.Sequential(
            nn.Linear(dy, my),
            nn.Tanh(),
            nn.Linear(my, my),
            nn.Tanh(),
            nn.Linear(my, myout)
        )
        net_y = net_y.to(prec).to(device)
        net_x = Phi(nTh=nTh, m=m, dx=dx, dy=myout, alph=alpha)
        net_x = net_x.to(prec).to(device)

        # ADAM optimizer
        optim = torch.optim.Adam(list(net_x.parameters()) + list(net_y.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)

        net_x.train()
        for epoch in range(num_epochs):
            # train
            for xy in train_loader:
                xy = cvt(xy)
                x = xy[:, :100].view(-1, 100) @ Vx
                y = xy[:, 100:].view(-1, dy)

                # update network for pi(x|y)
                for step in range(num_steps):
                    # update network for pi(x|y)
                    optim.zero_grad()
                    u = net_y(y)
                    loss_x, costs_x = compute_loss(net_x, x, u, nt=nt)
                    loss_x.backward()
                    optim.step()
                    for p in net_x.parameters():
                        p.data = torch.clamp(p.data, clampMin, clampMax)

                    if torch.isnan(loss_x):  # catch NaNs when hyperparameters are poorly chosen
                        logger.info("NaN encountered....exiting prematurely")
                        exit(1)
                # end batch_iter

        net_x.eval()
        valAlphMeterCx = utils.AverageMeter()
        with torch.no_grad():
            for xy_valid in valid_loader:
                xy_valid = cvt(xy_valid)
                nex = xy_valid.shape[0]
                x_valid = xy_valid[:, :100].view(-1, 100) @ Vx
                y_valid = xy_valid[:, 100:].view(-1, dy)
                _, val_costs_x = compute_loss(net_x, x_valid, net_y(y_valid), nt=nt_val)
                val_costs_Cx = val_costs_x[1]
                valAlphMeterCx.update(val_costs_Cx.item(), nex)
            Cx = valAlphMeterCx.avg

        log_message = '{:05d}  {:9.3e}  '.format(trial + 1, Cx)
        logger.info(log_message)
        valid_hist.loc[len(valid_hist.index)] = [Cx]

    params_hist.to_csv(os.path.join(args.save, '%s_params_hist.csv' % args.data))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % args.data))

