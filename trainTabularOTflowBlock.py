import argparse
import os
import pandas as pd
import time
import numpy as np
import datetime
import lib.utils as utils
from lib.utils import count_parameters
from torch.utils.data import DataLoader
from datasets import tabulardata
from src.mmd import mmd
from src.OTFlowProblem import *
from src.Phi import *
from lib.tabloader import tabloader

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['wt_wine', 'rd_wine', 'parkinson'], type=str, default='rd_wine'
)

parser.add_argument("--nt", type=int, default=6, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=10, help="number of time steps for validation")
parser.add_argument('--alph', type=str, default='1.0,100.0,15.0')
parser.add_argument('--m', type=int, default=256)
parser.add_argument('--nTh', type=int, default=2)
parser.add_argument('--dx', type=int, default=6, help="number of dimensions for x")

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument("--drop_freq", type=int, default=0,
                    help="how often to decrease learning rate; 0 lets the mdoel choose")
parser.add_argument("--lr_drop", type=float, default=2.0, help="how much to decrease learning rate (divide by)")
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--prec', type=str, default='single', choices=['single', 'double'],
                    help="single or double precision")
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--test_ratio', type=int, default=0.10)
parser.add_argument('--valid_ratio', type=int, default=0.10)
parser.add_argument('--random_state', type=int, default=42)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--early_stopping', type=int, default=10)

parser.add_argument('--save', type=str, default='experiments/cnf/tabjoint')
parser.add_argument('--val_freq', type=int,
                    default=20)  # validation frequency needs to be less than viz_freq or equal to viz_freq
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

if args.prec == 'double':
    prec = torch.float64
else:
    prec = torch.float32

# decrease the learning rate based on validation
ndecs_nety = 0
n_vals_wo_improve_nety = 0


def update_lr_nety(optimizer, n_vals_without_improvement):
    global ndecs_nety
    if ndecs_nety == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs_nety = 1
    elif ndecs_nety == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop ** 2
        ndecs_nety = 2
    else:
        ndecs_nety += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop ** ndecs_nety


ndecs_netx = 0
n_vals_wo_improve_netx = 0


def update_lr_netx(optimizer, n_vals_without_improvement):
    global ndecs_netx
    if ndecs_netx == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs_netx = 1
    elif ndecs_netx == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop ** 2
        ndecs_netx = 2
    else:
        ndecs_netx += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop ** ndecs_netx


def compute_loss(net, x, y, nt):
    Jc, cs = OTFlowProblem(x, y, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


def load_data(dataset):
    if dataset == 'wt_wine':
        data = tabulardata.get_wt_wine()
    elif dataset == 'rd_wine':
        data = tabulardata.get_rd_wine()
    elif dataset == 'parkinson':
        data = tabulardata.get_parkinson()
    else:
        raise Exception("Dataset is Incorrect")
    return data


def evaluate_model(nety, netx, data, batch_size, test_ratio, valid_ratio, random_state, dx, nt_val, prec, bestParams_y,
                   bestParams_x):
    _, _, testData, _ = tabloader(data, batch_size, test_ratio, valid_ratio, random_state)
    testLoader = DataLoader(
        testData,
        batch_size=batch_size, shuffle=True
    )
    d = testData.shape[1]
    dy = d - dx
    nt_test = nt_val
    # reload model
    nety.load_state_dict(bestParams_y)
    nety = net_y.to(device)
    netx.load_state_dict(bestParams_x)
    netx = netx.to(device)
    # if specified precision supplied, override the loaded precision
    if prec != 'None':
        if prec == 'single':
            argPrec = torch.float32
        if prec == 'double':
            argPrec = torch.float64
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    nety.eval()
    netx.eval()

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
            tst_loss_y, tst_costs_y = compute_loss(nety, y_test, None, nt=nt_test)
            tst_loss_x, tst_costs_x = compute_loss(netx, x_test, y_test, nt=nt_test)
            total_lost = tst_loss_y + tst_loss_x
            total_L = tst_costs_y[0] + tst_costs_x[0]
            total_C = tst_costs_y[1] + tst_costs_x[1]
            total_R = tst_costs_y[2] + tst_costs_x[2]
            testLossMeter.update(total_lost.item(), nex_batch)
            testAlphMeterL.update(total_L.item(), nex_batch)
            testAlphMeterC.update(total_C.item(), nex_batch)
            testAlphMeterR.update(total_R.item(), nex_batch)

        # generate samples
        dat = load_data(data)
        dat = tabulardata.process_data(dat)
        dat = tabulardata.normalize_data(dat)
        dat = torch.tensor(dat, dtype=torch.float32)
        normSamples = torch.randn(dat.shape[0], dat.shape[1]).to(device)
        modelGen = np.zeros(normSamples.shape)
        zx = normSamples[:, dy:].view(-1, dx)
        zy = normSamples[:, :dy].view(-1, dy)
        finvy = integrate(zy, None, nety, [1.0, 0.0], nt_test, stepper="rk4", alph=nety.alph)
        finvx = integrate(zx, finvy[:, :dy], netx, [1.0, 0.0], nt_test, stepper="rk4", alph=netx.alph)
        modelGen[:, :dy] = finvy[:, :dy].detach().cpu().numpy()
        modelGen[:, dy:] = finvx[:, :dx].detach().cpu().numpy()

        return testAlphMeterC.avg, mmd(modelGen, dat)


if __name__ == '__main__':

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    train_loader, valid_loader, test_data, train_size = tabloader(args.data, args.batch_size, args.test_ratio,
                                                                  args.valid_ratio, args.random_state)

    # hyperparameters of model
    d = test_data.shape[1]
    dx = args.dx
    dy = d - dx

    alph = args.alph
    nt = args.nt
    nt_val = args.nt_val
    nTh = args.nTh
    m = args.m

    # set up neural network to model potential function Phi
    net_y = Phi(nTh=nTh, m=args.m, dx=dy, dy=0, alph=alph)
    net_y = net_y.to(prec).to(device)
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
        net_y = Phi(nTh=nTh, m=m, dx=dy, dy=0, alph=alph)
        net_x = Phi(nTh=nTh, m=m, dx=dx, dy=dy, alph=alph)

        prec = checkpt['state_dict']['A'].dtype
        net_y = net_y.to(prec)
        net_x = net_x.to(prec)
        net_y.load_state_dict(checkpt["state_dict_y"])
        net_y = net_y.to(device)
        net_x.load_state_dict(checkpt["state_dict_x"])
        net_x = net_x.to(device)

    if args.val_freq == 0:
        # if val_freq set to 0, then validate after every epoch
        args.val_freq = math.ceil(train_size / args.batch_size)

    # ADAM optimizer
    optim_y = torch.optim.Adam(net_y.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=0.04 good
    optim_x = torch.optim.Adam(net_x.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=0.04 good

    strTitle = args.data + '_' + start_time

    logger.info(net_y)
    logger.info(net_x)
    logger.info("-------------------------")
    logger.info("dx={:} dy={:}  m={:}  nTh={:}   alpha={:}".format(dx, dy, m, nTh, alph))
    logger.info("nt={:}   nt_val={:}".format(nt, nt_val))
    logger.info("Number of trainable parameters for y: {}".format(count_parameters(net_y)))
    logger.info("Number of trainable parameters for x: {}".format(count_parameters(net_x)))
    logger.info("-------------------------")
    logger.info(str(optim_y))  # optimizer info
    logger.info(str(optim_x))  # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxEpochs={:} val_freq={:}".format(args.num_epochs, args.val_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    columns_train = ["step", "train_loss_x", "train_loss_y", "train_Lx", "train_Ly", "train_Cx", "train_Cy",
                     "train_Rx", "train_Ry"]
    columns_valid = ["valid_loss_x", "valid_loss_y", "valid_Lx", "valid_Ly", "valid_Cx", "valid_Cy", "valid_Rx",
                     "valid_Ry"]
    train_hist = pd.DataFrame(columns=columns_train)
    valid_hist = pd.DataFrame(columns=columns_valid)

    begin = time.time()
    end = begin
    best_loss_nety = float('inf')
    best_loss_netx = float('inf')
    best_cs_nety = [0.0] * 3
    best_cs_netx = [0.0] * 3
    bestParams_nety = None
    bestParams_netx = None
    total_itr = (int(train_size / args.batch_size) + 1) * args.num_epochs

    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s}   {:9s}  {:9s}  {:9s}  {:9s}   {:9s}  {:9s}  {:9s}  {:9s}'.format(
            'iter', ' time', 'loss_y', 'loss_x', 'Ly (L2)', 'Lx (L2)', 'Cy (nll)', 'Cx (nll)', 'Ry (HJB)',
            'Rx (HJB)', 'valLoss', 'valL', 'valC', 'valR'
        )
    )
    logger.info(log_msg)

    time_meter = utils.AverageMeter()

    # box constraints / acceptable range for parameter values
    clampMax = 1.5
    clampMin = -1.5

    net_y.train()
    net_x.train()
    itr = 1
    for epoch in range(args.num_epochs):
        # train
        for _, xy in enumerate(train_loader):
            xy = cvt(xy)
            x = xy[:, dy:].view(-1, dx)
            y = xy[:, :dy].view(-1, dy)

            # update network for pi(y)
            optim_y.zero_grad()
            for p in net_y.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)

            loss_y, costs_y = compute_loss(net_y, y, None, nt=nt)
            loss_y.backward()
            optim_y.step()

            # update network for pi(x|y)
            optim_x.zero_grad()
            for p in net_x.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)
            loss_x, costs_x = compute_loss(net_x, x, y, nt=nt)
            loss_x.backward()
            optim_x.step()

            time_meter.update(time.time() - end)

            log_message = (
                '{:05d}   {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                    itr, time_meter.val, loss_y, loss_x, costs_y[0], costs_x[0], costs_y[1], costs_x[1], costs_y[2],
                    costs_x[2]
                )
            )
            loss = loss_y + loss_x
            if torch.isnan(loss):  # catch NaNs when hyperparameters are poorly chosen
                logger.info(log_message)
                logger.info("NaN encountered....exiting prematurely")
                logger.info("Training Time: {:} seconds".format(time_meter.sum))
                logger.info('File: ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                            f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
                exit(1)

            train_hist.loc[len(train_hist.index)] = [itr, loss_x.item(), loss_y.item(), costs_x[0].item(),
                                                     costs_y[0].item(),
                                                     costs_x[1].item(), costs_y[1].item(), costs_x[2].item(),
                                                     costs_y[2].item()]

            # validation
            if itr % args.val_freq == 0 or itr == total_itr:
                net_y.eval()
                net_x.eval()
                with torch.no_grad():

                    valLossMetery = utils.AverageMeter()
                    valAlphMeterLy = utils.AverageMeter()
                    valAlphMeterCy = utils.AverageMeter()
                    valAlphMeterRy = utils.AverageMeter()
                    valLossMeterx = utils.AverageMeter()
                    valAlphMeterLx = utils.AverageMeter()
                    valAlphMeterCx = utils.AverageMeter()
                    valAlphMeterRx = utils.AverageMeter()

                    for _, xy_valid in enumerate(valid_loader):
                        xy_valid = cvt(xy_valid)
                        nex = xy_valid.shape[0]
                        x_valid = xy_valid[:, dy:].view(-1, dx)
                        y_valid = xy_valid[:, :dy].view(-1, dy)

                        val_loss_y, val_costs_y = compute_loss(net_y, y_valid, None, nt=nt_val)
                        val_loss_x, val_costs_x = compute_loss(net_x, x_valid, y_valid, nt=nt_val)

                        valLossMetery.update(val_loss_y.item(), nex)
                        valLossMeterx.update(val_loss_x.item(), nex)

                        val_costs_Ly = val_costs_y[0]
                        val_costs_Cy = val_costs_y[1]
                        val_costs_Ry = val_costs_y[2]

                        val_costs_Lx = val_costs_x[0]
                        val_costs_Cx = val_costs_x[1]
                        val_costs_Rx = val_costs_x[2]

                        valAlphMeterLy.update(val_costs_Ly.item(), nex)
                        valAlphMeterCy.update(val_costs_Cy.item(), nex)
                        valAlphMeterRy.update(val_costs_Ry.item(), nex)
                        valAlphMeterLx.update(val_costs_Lx.item(), nex)
                        valAlphMeterCx.update(val_costs_Cx.item(), nex)
                        valAlphMeterRx.update(val_costs_Rx.item(), nex)

                    valid_hist.loc[len(valid_hist.index)] = [valLossMeterx.avg, valLossMetery.avg, valAlphMeterLx.avg,
                                                             valAlphMeterLy.avg, valAlphMeterCx.avg, valAlphMeterCy.avg,
                                                             valAlphMeterRx.avg, valAlphMeterRy.avg]
                    # add to print message
                    log_message += '  {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}  '.format(
                        valLossMetery.avg + valLossMeterx.avg, valAlphMeterLy.avg + valAlphMeterLx.avg,
                        valAlphMeterCy.avg + valAlphMeterCx.avg, valAlphMeterRy.avg + valAlphMeterRx.avg
                    )

                    # save best set of parameters
                    if valLossMetery.avg < best_loss_nety:
                        n_vals_wo_improve_nety = 0
                        best_loss_nety = valLossMetery.avg
                        best_cs_nety = [valAlphMeterLy.avg, valAlphMeterCy.avg, valAlphMeterRy.avg]
                        utils.makedirs(args.save)
                        bestParams_y = net_y.state_dict()
                        bestParams_x = net_x.state_dict()

                        torch.save({
                            'args': args,
                            'state_dict_y': bestParams_y,
                            'state_dict_x': bestParams_x,
                        }, os.path.join(args.save,
                                        start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                                        f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth'))
                    else:
                        n_vals_wo_improve_nety += 1
                    log_message += 'nety no improve: {:d}/{:d}  '.format(n_vals_wo_improve_nety, args.early_stopping)

                    if valLossMeterx.avg < best_loss_netx:
                        n_vals_wo_improve_netx = 0
                        best_loss_netx = valLossMeterx.avg
                        best_cs_netx = [valAlphMeterLx.avg, valAlphMeterCx.avg, valAlphMeterRx.avg]
                        utils.makedirs(args.save)
                        bestParams_y = net_y.state_dict()
                        bestParams_x = net_x.state_dict()

                        torch.save({
                            'args': args,
                            'state_dict_y': bestParams_y,
                            'state_dict_x': bestParams_x,
                        }, os.path.join(args.save,
                                        start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                                        f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth'))
                    else:
                        n_vals_wo_improve_netx += 1
                    log_message += 'netx no improve: {:d}/{:d}'.format(n_vals_wo_improve_netx, args.early_stopping)

                    net_y.train()
                    net_x.train()
            logger.info(log_message)  # print iteration

            if args.drop_freq == 0:  # if set to the code setting 0 , the lr drops based on validation
                if n_vals_wo_improve_nety > args.early_stopping:
                    if ndecs_nety > 2:
                        logger.info("early stopping engaged")
                        logger.info("Training Time: {:} seconds".format(time_meter.sum))
                        logger.info(
                            'File: ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                            f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
                        NLL, MMD = evaluate_model(net_y, net_x, args.data, args.batch_size, args.test_ratio,
                                                  args.valid_ratio, args.random_state, args.dx, args.nt_val,
                                                  args.prec, bestParams_y, bestParams_x)
                        columns_test = ["alpha", "batch_size", "lr", "width", "nt", "NLL", "MMD", "time", "iter"]
                        test_hist = pd.DataFrame(columns=columns_test)
                        test_hist.loc[len(test_hist.index)] = [args.alph, args.batch_size, args.lr, args.m, args.nt,
                                                               NLL, MMD, time_meter.sum, itr]
                        testfile_name = '/local/scratch3/zwan736/OT-Flow/experiments/cnf/tabjoint/' + args.data + '_test_hist.csv'
                        if os.path.isfile(testfile_name):
                            test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
                        else:
                            test_hist.to_csv(testfile_name, index=False)
                        exit(0)
                    else:
                        update_lr_nety(optim_y, n_vals_wo_improve_nety)
                        n_vals_wo_improve_nety = 0

                if n_vals_wo_improve_netx > args.early_stopping:
                    if ndecs_netx > 2:
                        logger.info("early stopping engaged")
                        logger.info("Training Time: {:} seconds".format(time_meter.sum))
                        logger.info(
                            'File: ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                            f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
                        train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                        valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                        NLL, MMD = evaluate_model(net_y, net_x, args.data, args.batch_size, args.test_ratio,
                                                  args.valid_ratio, args.random_state, args.dx, args.nt_val,
                                                  args.prec, bestParams_y, bestParams_x)
                        columns_test = ["alpha", "batch_size", "lr", "width", "nt", "NLL", "MMD", "time", "iter"]
                        test_hist = pd.DataFrame(columns=columns_test)
                        test_hist.loc[len(test_hist.index)] = [args.alph, args.batch_size, args.lr, args.m, args.nt,
                                                               NLL, MMD, time_meter.sum, itr]
                        testfile_name = '/local/scratch3/zwan736/OT-Flow/experiments/cnf/tabjoint/' + args.data + '_test_hist.csv'
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
                    for p in optim_y.param_groups:
                        p['lr'] /= args.lr_drop
                    for p in optim_x.param_groups:
                        p['lr'] /= args.lr_drop
                    print("lr: ", p['lr'])

            itr += 1
            end = time.time()
            # end batch_iter

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' + \
                f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
    NLL, MMD = evaluate_model(net_y, net_x, args.data, args.batch_size, args.test_ratio,
                              args.valid_ratio, args.random_state, args.dx, args.nt_val,
                              args.prec, bestParams_y, bestParams_x)
    columns_test = ["alpha", "batch_size", "lr", "width", "nt", "NLL", "MMD", "time", "iter"]
    test_hist = pd.DataFrame(columns=columns_test)
    test_hist.loc[len(test_hist.index)] = [args.alph, args.batch_size, args.lr, args.m, args.nt, NLL, MMD,
                                           time_meter.sum, itr]
    testfile_name = '/local/scratch3/zwan736/OT-Flow/experiments/cnf/tabjoint/' + args.data + '_test_hist.csv'
    if os.path.isfile(testfile_name):
        test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
    else:
        test_hist.to_csv(testfile_name, index=False)

