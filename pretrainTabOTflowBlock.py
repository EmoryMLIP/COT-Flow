import argparse
import os
import datetime
import numpy as np
import pandas as pd
import lib.utils as utils

from src.OTFlowProblem import *
from src.Phi import *
from lib.tabloader import tabloader

parser = argparse.ArgumentParser('COT-Flow')
parser.add_argument(
    '--data', choices=['wt_wine', 'rd_wine', 'parkinson'], type=str, default='rd_wine'
)

parser.add_argument("--nt_val", type=int, default=32, help="number of time steps for validation")
parser.add_argument('--nTh'   , type=int, default=2)
parser.add_argument('--dx'   , type=int, default=6, help="number of dimensions for x")

parser.add_argument("--drop_freq", type=int  , default=0, help="how often to decrease learning rate; 0 lets the mdoel choose")
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--prec'      , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--num_epochs'    , type=int, default=3)
parser.add_argument('--test_ratio', type=int, default=0.10)
parser.add_argument('--valid_ratio', type=int, default=0.10)
parser.add_argument('--random_state', type=int, default=42)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')

parser.add_argument('--save', type=str, default='experiments/cnf/tabjoint')
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

if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32


def compute_loss(net, x,y, nt):
    Jc , cs = OTFlowProblem(x, y, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    columns_params = ["alpha1", "alpha2", "nt", "width", "lr", "batchsz"]
    columns_valid = ["cx", "cy", "c"]
    params_hist = pd.DataFrame(columns=columns_params)
    valid_hist = pd.DataFrame(columns=columns_valid)

    log_msg = ('{:5s}  {:9s}  {:9s}  {:9s}'.format('trial', ' valCx', 'valCy', 'valC'))
    logger.info(log_msg)

    # box constraints / acceptable range for parameter values
    clampMax = 1.5
    clampMin = -1.5

    # sample space for hyperparameters
    width_list = np.array([32, 64, 128, 256, 512])
    batch_size_list = np.array([32, 64])
    lr_list = np.array([0.01, 0.005, 0.001])
    nt_list = np.array([8, 16])

    for trial in range(50):

        batch_size = int(np.random.choice(batch_size_list))
        train_loader, valid_loader, test_data, train_size = tabloader(args.data, batch_size, args.test_ratio,
                                                                      args.valid_ratio, args.random_state)

        d = test_data.shape[1]
        dx = args.dx
        dy = d - dx

        width = np.random.choice(width_list)
        lr = np.random.choice(lr_list)
        nt = np.random.choice(nt_list)
        alpha = [1.0, np.exp(np.random.uniform(-1, 3)), np.exp(np.random.uniform(-1, 3))]

        params_hist.loc[len(params_hist.index)] = [alpha[1], alpha[2], nt, width, lr, batch_size]

        nt_val = args.nt_val
        nTh = args.nTh

        # set up neural network to model potential function Phi
        net_y = Phi(nTh=nTh, m=width, dx=dy, dy=0, alph=alpha)
        net_y = net_y.to(prec).to(device)
        net_x = Phi(nTh=nTh, m=width, dx=dx, dy=dy, alph=alpha)
        net_x = net_x.to(prec).to(device)

        # ADAM optimizer
        optim_y = torch.optim.Adam(net_y.parameters(), lr=lr, weight_decay=args.weight_decay)  # lr=0.04 good
        optim_x = torch.optim.Adam(net_x.parameters(), lr=lr, weight_decay=args.weight_decay)  # lr=0.04 good

        if args.data == 'parkinson' or args.data == 'wt_wine':
            num_epochs = args.num_epochs
        else:
            num_epochs = 4

        net_y.train()
        net_x.train()

        for epoch in range(num_epochs):
            # train
            for xy in train_loader:
                xy = cvt(xy)
                x = xy[:, dy:].view(-1, dx)
                y = xy[:, :dy].view(-1, dy)

                # update network for pi(y)
                optim_y.zero_grad()
                for p in net_y.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)

                loss_y,costs_y  = compute_loss(net_y, y, None, nt=nt)
                loss_y.backward()
                optim_y.step()

                # update network for pi(x|y)
                optim_x.zero_grad()
                for p in net_x.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)
                loss_x, costs_x = compute_loss(net_x, x, y, nt=nt)
                loss_x.backward()
                optim_x.step()

                loss = loss_y + loss_x
                if torch.isnan(loss): # catch NaNs when hyperparameters are poorly chosen
                    logger.info("NaN encountered....exiting prematurely")
                    exit(1)
                # end batch_iter

        valAlphMeterCy = utils.AverageMeter()
        valAlphMeterCx = utils.AverageMeter()

        for xy_valid in valid_loader:

            xy_valid = cvt(xy_valid)
            nex = xy_valid.shape[0]
            x_valid = xy_valid[:, dy:].view(-1, dx)
            y_valid = xy_valid[:, :dy].view(-1, dy)

            _, val_costs_y = compute_loss(net_y, y_valid, None, nt=nt_val)
            _, val_costs_x = compute_loss(net_x, x_valid, y_valid, nt=nt_val)

            val_costs_Cy = val_costs_y[1]
            val_costs_Cx = val_costs_x[1]

            valAlphMeterCx.update(val_costs_Cx.item(), nex)
            valAlphMeterCy.update(val_costs_Cy.item(), nex)

        Cx = valAlphMeterCx.avg
        Cy = valAlphMeterCy.avg
        C = Cx + Cy

        log_message = '{:05d}  {:9.3e}  {:9.3e}  {:9.3e}  '.format(trial+1, Cx, Cy, C)
        logger.info(log_message)
        valid_hist.loc[len(valid_hist.index)] = [Cx, Cy, C]

    params_hist.to_csv(os.path.join(args.save, '%s_params_hist.csv' % args.data))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % args.data))

