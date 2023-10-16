# trainOTflowCond.py
# train COT-Flow for small tabular datasets and the stochastic Lotka-Volterra problem
import argparse
import os
import pandas as pd
import scipy.io
import numpy as np
import time
import datetime
import lib.utils as utils
from lib.utils import count_parameters
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.OTFlowProblem import *
from src.Phi import *
from src.mmd import mmd
from lib.tabloader import tabloader

parser = argparse.ArgumentParser('COT-Flow')
parser.add_argument(
    '--data', choices=['concrete', 'energy', 'yacht', 'lv'], type=str, default='concrete'
)

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=32, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0,100.0,15.0')
parser.add_argument('--m'     , type=int, default=256)
parser.add_argument('--nTh'   , type=int, default=2)
parser.add_argument('--dx'   , type=int, default=1, help="number of dimensions for x")


parser.add_argument('--lr'       , type=float, default=0.01)
parser.add_argument("--drop_freq", type=int  , default=0, help="how often to decrease learning rate; 0 lets the mdoel choose")
parser.add_argument("--lr_drop"  , type=float, default=2.0, help="how much to decrease learning rate (divide by)")
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--prec'      , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--num_epochs'    , type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--test_ratio', type=int, default=0.10)
parser.add_argument('--valid_ratio', type=int, default=0.10)
parser.add_argument('--random_state', type=int, default=42)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--early_stopping', type=int, default=10)

parser.add_argument('--save', type=str, default='experiments/cnf/tabcond')
parser.add_argument('--save_test', type=int, default=1, help="if 1 evaluate after training if 0 not")
parser.add_argument('--val_freq', type=int, default=20) # validation frequency needs to be less than viz_freq or equal to viz_freq
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

# add timestamp to save path
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32


# decrease the learning rate based on validation
ndecs_netx = 0
n_vals_wo_improve_netx=0
def update_lr_netx(optimizer, n_vals_without_improvement):
    global ndecs_netx
    if ndecs_netx == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs_netx = 1
    elif ndecs_netx == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**2
        ndecs_netx = 2
    else:
        ndecs_netx += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs_netx


def load_data(data, test_ratio, valid_ratio, batch_size, random_state):

    if data == 'lv':
        # TODO change to correct path
        dataset_load = scipy.io.loadmat('.../OT-Flow/datasets/training_data.mat')
        x_train = dataset_load['x_train']
        y_train = dataset_load['y_train']
        dataset = np.concatenate((x_train, y_train), axis=1)
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


def compute_loss(net, x,y, nt):
    Jc , cs = OTFlowProblem(x, y, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


def evaluate_model(net, data, batch_size, test_ratio, valid_ratio, random_state, dx, nt_val, prec, bestParams_x):

    _, _, testData, _ = tabloader(data, batch_size, test_ratio, valid_ratio, random_state)
    testLoader = DataLoader(
        testData,
        batch_size=batch_size, shuffle=True
    )
    d = testData.shape[1]
    dy = d - dx
    nt_test = nt_val
    # reload model
    net.load_state_dict(bestParams_x)
    net = net.to(device)
    # if specified precision supplied, override the loaded precision
    if prec != 'None':
        if prec == 'single':
            argPrec = torch.float32
        if prec == 'double':
            argPrec = torch.float64

    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)
    net.eval()
    with torch.no_grad():
        # meters to hold testing results
        testLossMeter = utils.AverageMeter()
        testAlphMeterL = utils.AverageMeter()
        testAlphMeterC = utils.AverageMeter()
        testAlphMeterR = utils.AverageMeter()
        for _, x0 in enumerate(testLoader):
            x0 = cvt(x0)
            nex_batch = x0.shape[0]
            x_test = x0[:, dy:].view(-1, dx)
            y_test = x0[:, :dy].view(-1, dy)
            tst_loss_x, tst_costs_x = compute_loss(net, x_test, y_test, nt=nt_test)
            testLossMeter.update(tst_loss_x.item(), nex_batch)
            testAlphMeterL.update(tst_costs_x[0].item(), nex_batch)
            testAlphMeterC.update(tst_costs_x[1].item(), nex_batch)
            testAlphMeterR.update(tst_costs_x[2].item(), nex_batch)

        # generate samples
        normSamples = torch.randn(testData.shape[0], 1).to(device)
        zx = cvt(normSamples)
        finvx = integrate(zx, testData[:, :dy].to(device), net, [1.0, 0.0], nt_test, stepper="rk4", alph=net.alph)
        modelGen = finvx[:, :dx].detach().cpu().numpy()
        # compute MMD

        return testAlphMeterC.avg, mmd(modelGen, testData[:, dy:])


if __name__ == '__main__':

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    # load data
    train_loader, valid_loader, n_train, n_feat = load_data(args.data, args.test_ratio, args.valid_ratio,
                                                    args.batch_size, args.random_state)

    # hyperparameters of model
    d   = n_feat
    dx = args.dx
    dy = d -dx

    alph = args.alph
    nt  = args.nt
    nt_val = args.nt_val
    nTh = args.nTh
    m   = args.m

    # set up neural network to model potential function Phi
    net_x = Phi(nTh=nTh, m=args.m, dx=dx, dy=dy, alph=alph)
    net_x = net_x.to(prec).to(device)

    # resume training on a model that's already had some training
    if args.resume is not None:
        # reload model
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        m = checkpt['args'].m
        alph = args.alph  # overwrite saved alpha
        nTh = checkpt['args'].nTh
        args.hutch = checkpt['args'].hutch
        prec = checkpt['state_dict']['A'].dtype
        net_x = Phi(nTh=nTh, m=m, dx=dx, dy=dy, alph=alph)
        net_x = net_x.to(prec)
        net_x.load_state_dict(checkpt["state_dict_x"])
        net_x = net_x.to(device)

    if args.val_freq == 0:
        # if val_freq set to 0, then validate after every epoch
        args.val_freq = math.ceil(n_train/args.batch_size)

    # ADAM optimizer
    optim_x = torch.optim.Adam(net_x.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=0.04 good

    strTitle = args.data + '_' + start_time

    logger.info(net_x)
    logger.info("-------------------------")
    logger.info("dx={:} dy={:}  m={:}  nTh={:}   alpha={:}".format(dx, dy, m, nTh, alph))
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
    logger.info("Number of trainable parameters for x: {}".format(count_parameters(net_x)))
    logger.info("-------------------------")
    logger.info(str(optim_x))  # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxEpochs={:} val_freq={:}".format(args.num_epochs, args.val_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    columns_train = ["step", "train_loss_x", "train_L", "train_C", "train_R"]
    columns_valid = ["valid_loss_x", "valid_L", "valid_C", "valid_R"]
    train_hist = pd.DataFrame(columns=columns_train)
    valid_hist = pd.DataFrame(columns=columns_valid)

    begin = time.time()
    end = begin
    best_loss_netx = float('inf')
    best_cs_netx = [0.0] * 3
    bestParams_netx = None
    total_itr = (int(n_train / args.batch_size) + 1) * args.num_epochs

    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}   {:9s}  {:9s}   {:9s}  {:9s}  {:9s}  {:9s}'.format(
            'iter', ' time', 'loss_x', 'Lx (L2)', 'Cx (nll)', 'Rx (HJB)', 'valLoss', 'valL', 'valC', 'valR'
        )
    )
    logger.info(log_msg)

    time_meter = utils.AverageMeter()

    # box constraints / acceptable range for parameter values
    clampMax = 1.5
    clampMin = -1.5

    net_x.train()
    itr = 1
    for epoch in range(args.num_epochs):
        # train
        for _, xy in enumerate(train_loader):
            xy = cvt(xy)
            if args.data == 'lv':
                x = xy[:, :dx].view(-1, dx)
                y = xy[:, dx:].view(-1, dy)
            else:
                x = xy[:, dy:].view(-1, dx)
                y = xy[:, :dy].view(-1, dy)

            # update network for pi(x|y)
            end = time.time()

            optim_x.zero_grad()
            for p in net_x.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)
            loss_x, costs_x = compute_loss(net_x, x, y, nt=nt)
            loss_x.backward()
            optim_x.step()

            time_meter.update(time.time() - end)

            log_message = (
                '{:05d}   {:6.3f}   {:9.3e}  {:9.3e}   {:9.3e}  {:9.3e} '.format(
                    itr, time_meter.val, loss_x, costs_x[0], costs_x[1], costs_x[2]
                )
            )
            if torch.isnan(loss_x): # catch NaNs when hyperparameters are poorly chosen
                logger.info(log_message)
                logger.info("NaN encountered....exiting prematurely")
                logger.info("Training Time: {:} seconds".format(time_meter.sum))
                logger.info('File: ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                            f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
                exit(1)

            train_hist.loc[len(train_hist.index)] = [itr, loss_x.item(), costs_x[0].item(), costs_x[1].item(),
                                                     costs_x[2].item()]
            # validation
            if itr % args.val_freq == 0 or itr == total_itr:
                net_x.eval()
                with torch.no_grad():
                    valLossMeterx = utils.AverageMeter()
                    valAlphMeterLx = utils.AverageMeter()
                    valAlphMeterCx = utils.AverageMeter()
                    valAlphMeterRx = utils.AverageMeter()
                    for xy_valid in valid_loader:
                        xy_valid = cvt(xy_valid)
                        nex = xy_valid.shape[0]
                        if args.data == 'lv':
                            x_valid = xy_valid[:, :dx].view(-1, dx)
                            y_valid = xy_valid[:, dx:].view(-1, dy)
                        else:
                            x_valid = xy_valid[:, dy:].view(-1, dx)
                            y_valid = xy_valid[:, :dy].view(-1, dy)

                        val_loss_x, val_costs_x = compute_loss(net_x, x_valid, y_valid, nt=nt_val)
                        valLossMeterx.update(val_loss_x.item(), nex)

                        val_costs_Lx = val_costs_x[0]
                        val_costs_Cx = val_costs_x[1]
                        val_costs_Rx = val_costs_x[2]

                        valAlphMeterLx.update(val_costs_Lx.item(), nex)
                        valAlphMeterCx.update(val_costs_Cx.item(), nex)
                        valAlphMeterRx.update(val_costs_Rx.item(), nex)
                    Loss = valLossMeterx.avg
                    Lx = valAlphMeterLx.avg
                    Cx = valAlphMeterCx.avg
                    Rx = valAlphMeterRx.avg

                    valid_hist.loc[len(valid_hist.index)] = [Loss, Lx, Cx, Rx]
                    # add to print message
                    log_message += '  {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}  '.format(Loss, Lx, Cx, Rx)

                    # save best set of parameters
                    if Loss < best_loss_netx:
                        n_vals_wo_improve_netx = 0
                        best_loss_netx = Loss
                        best_cs_netx = [Lx, Cx, Rx]
                        utils.makedirs(args.save)
                        bestParams_x = net_x.state_dict()
                        torch.save({
                            'args': args,
                            'state_dict_x': bestParams_x,
                        }, os.path.join(args.save,
                                        start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                                        f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth'))
                    else:
                        n_vals_wo_improve_netx += 1
                    log_message += 'netx no improve: {:d}/{:d}'.format(n_vals_wo_improve_netx, args.early_stopping)

                    net_x.train()
            logger.info(log_message) # print iteration

            if args.drop_freq == 0:  # if set to the code setting 0 , the lr drops based on validation
                if n_vals_wo_improve_netx > args.early_stopping:
                    if ndecs_netx > 2:
                        logger.info("early stopping engaged")
                        logger.info("Training Time: {:} seconds".format(time_meter.sum))
                        logger.info('File: ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                                    f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
                        train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                        valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                        if bool(args.save_test) is False:
                            exit(0)
                        NLL, MMD = evaluate_model(net_x, args.data, args.batch_size, args.test_ratio, args.valid_ratio,
                                                  args.random_state, args.dx, args.nt_val, args.prec, bestParams_x)
                        columns_test = ["alpha", "batch_size", "lr", "width", "nt", "NLL", "MMD", "time", "iter"]
                        test_hist = pd.DataFrame(columns=columns_test)
                        test_hist.loc[len(test_hist.index)] = [args.alph, args.batch_size, args.lr, args.m, args.nt,
                                                               NLL, MMD, time_meter.sum, itr]
                        testfile_name = '.../OT-Flow/experiments/cnf/tabcond/' + args.data + '_test_hist.csv'
                        if os.path.isfile(testfile_name):
                            test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
                        else:
                            test_hist.to_csv(testfile_name, index=False)
                        exit(0)
                    else:
                        update_lr_netx(optim_x, n_vals_wo_improve_netx)
                        n_vals_wo_improve_netx = 0
            else:
                # shrink step size
                if itr % args.drop_freq == 0:
                    for p in optim_x.param_groups:
                        p['lr'] /= args.lr_drop
                    print("lr: ", p['lr'])

            itr += 1
            # end batch_iter

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
    if bool(args.save_test) is False:
        exit(0)
    NLL, MMD = evaluate_model(net_x, args.data, args.batch_size, args.test_ratio, args.valid_ratio, args.random_state,
                              args.dx, args.nt_val, args.prec, bestParams_x)
    columns_test = ["alpha", "batch_size", "lr", "width", "nt", "NLL", "MMD", "time", "iter"]
    test_hist = pd.DataFrame(columns=columns_test)
    test_hist.loc[len(test_hist.index)] = [args.alph, args.batch_size, args.lr, args.m, args.nt, NLL, MMD,
                                           time_meter.sum, itr]
    testfile_name = '.../OT-Flow/experiments/cnf/tabcond/' + args.data + '_test_hist.csv'
    if os.path.isfile(testfile_name):
        test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
    else:
        test_hist.to_csv(testfile_name, index=False)

