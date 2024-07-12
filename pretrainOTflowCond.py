import argparse
import os
import datetime
import numpy as np
import pandas as pd
import lib.utils as utils
import scipy.io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.OTFlowProblem import *
from src.Phi import *
from lib.tabloader import tabloader

parser = argparse.ArgumentParser('COT-Flow')
parser.add_argument(
    '--data', choices=['concrete', 'energy', 'yacht', 'lv'], type=str, default='concrete'
)

parser.add_argument("--nt_val", type=int, default=32, help="number of time steps for validation")
parser.add_argument('--nTh', type=int, default=2)
parser.add_argument('--dx', type=int, default=1, help="number of dimensions for x")

parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--prec', type=str, default='single', choices=['single', 'double'],
                    help="single or double precision")
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--num_trials', type=int, default=100, help="pilot run number of trials")
parser.add_argument('--test_ratio', type=int, default=0.10)
parser.add_argument('--valid_ratio', type=int, default=0.10)
parser.add_argument('--random_state', type=int, default=42)

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


def load_data(data, test_ratio, valid_ratio, batch_size, random_state):
    if data == 'lv':
        dataset_load = scipy.io.loadmat('.../COT-Flow/datasets/lv_data.mat')
        x_train = dataset_load['x_train']
        y_train = dataset_load['y_train']
        dataset = np.concatenate((x_train, y_train), axis=1)
        # log transformation over theta
        dataset[:, :4] = np.log(dataset[:, :4])

        # split data and convert to tensor
        train, valid = train_test_split(
            dataset, test_size=valid_ratio,
            random_state=random_state
        )
        train_sz = train.shape[0]
        feat_sz = train.shape[1]

        train_mean = np.mean(train, axis=0, keepdims=True)
        train_std = np.std(train, axis=0, keepdims=True)
        train_data = (train - train_mean) / train_std
        valid_data = (valid - train_mean) / train_std

        # convert to tensor
        train_data = torch.tensor(train_data, dtype=torch.float32)
        valid_data = torch.tensor(valid_data, dtype=torch.float32)

        # load train data
        trn_loader = DataLoader(
            train_data,
            batch_size=batch_size, shuffle=True
        )
        vld_loader = DataLoader(
            valid_data,
            batch_size=batch_size, shuffle=True
        )
    else:
        trn_loader, vld_loader, test_set, train_sz = tabloader(data, batch_size, test_ratio, valid_ratio, random_state)
        feat_sz = test_set.shape[1]

    return trn_loader, vld_loader, train_sz, feat_sz


def compute_loss(net, x, y, nt):
    Jc, cs = OTFlowProblem(x, y, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    columns_params = ["alpha1", "alpha2", "nt", "width", "lr", "batchsz"]
    columns_valid = ["cx"]
    params_hist = pd.DataFrame(columns=columns_params)
    valid_hist = pd.DataFrame(columns=columns_valid)

    log_msg = ('{:5s}  {:9s}'.format('trial', ' valCx'))
    logger.info(log_msg)

    # box constraints / acceptable range for parameter values
    clampMax = 1.5
    clampMin = -1.5

    # sample space for hyperparameters
    width_list = np.array([32, 64, 128, 256, 512])
    if args.data == 'lv':
        batch_size_list = np.array([32, 64, 128, 256])
    else:
        batch_size_list = np.array([32, 64])
    lr_list = np.array([0.01, 0.005, 0.001])
    nt_list = np.array([8, 16])

    for trial in range(args.num_trials):

        batch_size = int(np.random.choice(batch_size_list))
        train_loader, valid_loader, _, n_feat = load_data(args.data, args.test_ratio, args.valid_ratio,
                                                          batch_size, args.random_state)

        d = n_feat
        dx = args.dx
        dy = d - dx

        width = np.random.choice(width_list)
        lr = np.random.choice(lr_list)
        nt = np.random.choice(nt_list)
        # nt = 16
        alpha = [1.0, np.exp(np.random.uniform(-1, 3)), np.exp(np.random.uniform(-1, 3))]

        params_hist.loc[len(params_hist.index)] = [alpha[1], alpha[2], nt, width, lr, batch_size]

        nt_val = args.nt_val
        nTh = args.nTh

        # set up neural network to model potential function Phi
        net_x = Phi(nTh=nTh, m=width, dx=dx, dy=dy, alph=alpha)
        net_x = net_x.to(prec).to(device)

        # ADAM optimizer
        optim_x = torch.optim.Adam(net_x.parameters(), lr=lr, weight_decay=args.weight_decay)  # lr=0.04 good

        if args.data == 'lv':
            num_epochs = 1
        else:
            num_epochs = args.num_epochs

        net_x.train()

        for epoch in range(num_epochs):
            # train
            for xy in train_loader:
                xy = cvt(xy)
                if args.data == 'lv':
                    x = xy[:, :dx].view(-1, dx)
                    y = xy[:, dx:].view(-1, dy)
                else:
                    x = xy[:, dy:].view(-1, dx)
                    y = xy[:, :dy].view(-1, dy)

                # update network for pi(x|y)
                optim_x.zero_grad()
                for p in net_x.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)
                loss_x, costs_x = compute_loss(net_x, x, y, nt=nt)
                loss_x.backward()
                optim_x.step()

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
                if args.data == 'lv':
                    x_valid = xy_valid[:, :dx].view(-1, dx)
                    y_valid = xy_valid[:, dx:].view(-1, dy)
                else:
                    x_valid = xy_valid[:, dy:].view(-1, dx)
                    y_valid = xy_valid[:, :dy].view(-1, dy)
                _, val_costs_x = compute_loss(net_x, x_valid, y_valid, nt=nt_val)
                val_costs_Cx = val_costs_x[1]
                valAlphMeterCx.update(val_costs_Cx.item(), nex)
            Cx = valAlphMeterCx.avg

        log_message = '{:05d}  {:9.3e}  '.format(trial + 1, Cx)
        logger.info(log_message)
        valid_hist.loc[len(valid_hist.index)] = [Cx]

    params_hist.to_csv(os.path.join(args.save, '%s_params_hist.csv' % args.data))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % args.data))

