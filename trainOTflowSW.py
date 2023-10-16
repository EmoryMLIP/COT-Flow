# trainOTflowCond_SW.py
# train COT-Flow for the 1D shallow water problem
import argparse
import os
import pandas as pd
import time
import datetime
import torch.nn as nn
import lib.utils as utils
from lib.utils import count_parameters
import matplotlib.pyplot as plt
from src.OTFlowProblem import *
from src.Phi import *
from datasets.shallow_water import load_swdata

parser = argparse.ArgumentParser('COT-Flow')

parser.add_argument('--data', type=str, default='sw')
parser.add_argument("--nt"    , type=int, default=4, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=32, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0,100.0,15.0')
parser.add_argument('--m'     , type=int, default=64)
parser.add_argument('--m_y'     , type=int, default=128)
parser.add_argument('--mout_y'     , type=int, default=64)
parser.add_argument('--nTh'   , type=int, default=2)
parser.add_argument('--dx'   , type=int, default=14, help="number of dimensions for x")
parser.add_argument('--lr'       , type=float, default=0.001)
parser.add_argument("--drop_freq", type=int  , default=0, help="how often to decrease learning rate; 0 lets the mdoel choose")
parser.add_argument("--lr_drop"  , type=float, default=2.0, help="how much to decrease learning rate (divide by)")
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--prec'      , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--num_epochs'    , type=int, default=1000)
parser.add_argument('--num_steps'    , type=int, default=1, help="number of training steps for each example")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--test_ratio', type=int, default=0.10)
parser.add_argument('--valid_ratio', type=int, default=0.10)
parser.add_argument('--random_state', type=int, default=42)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--early_stopping', type=int, default=10)

parser.add_argument('--save', type=str, default='experiments/cnf/tabcond')
parser.add_argument('--val_freq', type=int, default=20) # validation frequency needs to be less than viz_freq or equal to viz_freq
parser.add_argument('--viz_freq', type=int, default=100) # frequency to visualize conditional sampling
parser.add_argument('--gpu', type=int, default=0)


args = parser.parse_args()
args.alph = [float(item) for item in args.alph.split(',')]
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

if args.resume is not None:
    # check if args.remue exists and if not throw an error
    assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
    # load args from checkpoint file
    checkpt = torch.load(args.resume, map_location=torch.device('cpu'))
    # overwrite all args related to the network architectures
    overwrite_args = ['m', 'm_y', 'mout_y', 'nTh', 'dx']
    for item in overwrite_args:
        setattr(args, item, getattr(checkpt['args'], item))


# add timestamp to save path
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size

if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32

# decrease the learning rate based on validation
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
            param_group["lr"] = args.lr / args.lr_drop**2
        ndecs_netx = 2
    else:
        ndecs_netx += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs_netx


def compute_loss(net, x, y, nt):
    Jc, cs = OTFlowProblem(x, y, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    # load data
    train_loader, valid_loader, n_train, n_feat, train_mean, train_std, Vy = load_swdata(args.batch_size, full=False)

    # hyperparameters of model
    d = n_feat
    dx = args.dx
    if dx < 100:
        x_full = train_loader.dataset[:, :100]
        x_full = x_full.view(-1, 100)
        cov_x = x_full.T @ x_full
        L, V = torch.linalg.eigh(cov_x)
        # get the last dx columns in V
        Vx = cvt(V[:, -dx:])
        perc = 100*torch.sum(L[-dx:]) / torch.sum(L)
        logger.info('Percentage of variance explained by first %d components: %.2f' % (dx, perc))
    else:
        Vx = cvt(torch.eye(100))

    dy = d - 100
    logger.info('Problem size: n_train=%d, dx: %d, dy: %d' % (n_train, dx, dy))
    # print shape of valid_data
    logger.info('Number of validation samples: %s' % str(len(valid_loader.dataset)))
    alph = args.alph
    nt = args.nt
    nt_val = args.nt_val
    nTh = args.nTh
    m = args.m

    # set up neural network to model potential function Phi
    net_x = Phi(nTh=nTh, m=args.m, dx=dx, dy=args.mout_y, alph=alph)
    net_x = net_x.to(prec).to(device)
    if args.resume is not None:
        net_x.load_state_dict(checkpt["state_dict_x"])

    if args.val_freq == 0:
        # if val_freq set to 0, then validate after every epoch
        args.val_freq = math.ceil(n_train/args.batch_size)

    # ADAM optimizer
    # make 3 layer multi-layer perceptron to process y data
    if args.m_y > 0 and args.mout_y > 0:
        net_y = nn.Sequential(
            nn.Linear(dy, args.m_y),
            nn.Tanh(),
            nn.Linear(args.m_y, args.m_y),
            nn.Tanh(),
            nn.Linear(args.m_y, args.mout_y)
        )
        net_y = net_y.to(prec).to(device)
        if args.resume is not None:
            net_y.load_state_dict(checkpt["state_dict_y"])
        # make one optimizer for net_x and net_y
        optim_x = torch.optim.Adam(list(net_x.parameters()) + list(net_y.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    else:
        net_y = lambda y: y
        optim_x = torch.optim.Adam(net_x.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    strTitle = args.data + '_' + start_time

    logger.info(net_x)
    logger.info("-------------------------")
    logger.info("dx={:} dy={:}  m={:}  nTh={:}   alpha={:}".format(dx, dy, m, nTh, alph))
    logger.info("nt={:}   nt_val={:}".format(nt, nt_val))
    logger.info("Number of trainable parameters for x: {}".format(count_parameters(net_x) + count_parameters(net_y)))
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
    net_y.train()
    itr = 1
    flag = 0
    for epoch in range(args.num_epochs):
        # train
        if flag > 0:
            break
        for xy in train_loader:
            if flag > 0:
                break
            xy = cvt(xy)
            x = xy[:, :100].view(-1, 100) @ Vx
            y = xy[:, 100:].view(-1, dy)

            for step in range(args.num_steps):
                # update network for pi(x|y)
                end = time.time()

                optim_x.zero_grad()
                u = net_y(y)
                loss_x, costs_x = compute_loss(net_x, x, u, nt=nt)
                loss_x.backward()
                optim_x.step()
                for p in net_x.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)

                time_meter.update(time.time() - end)

                log_message = (
                    '{:05d}   {:6.3f}   {:9.3e}  {:9.3e}   {:9.3e}  {:9.3e} '.format(
                        itr, time_meter.val, loss_x, costs_x[0], costs_x[1], costs_x[2]
                    )
                )
                if torch.isnan(loss_x):  # catch NaNs when hyperparameters are poorly chosen
                    logger.info(log_message)
                    logger.info("NaN encountered....exiting prematurely")
                    logger.info("Training Time: {:} seconds".format(time_meter.sum))
                    logger.info('File: ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' +
                                 f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
                    exit(1)

                train_hist.loc[len(train_hist.index)] = [itr, loss_x.item(), costs_x[0].item(), costs_x[1].item(),
                                                         costs_x[2].item()]
                # validation
                if itr % args.val_freq == 0 or itr == total_itr:
                    net_x.eval()
                    net_y.eval()
                    with torch.no_grad():

                        valLossMeter = utils.AverageMeter()
                        valAlphMeterL = utils.AverageMeter()
                        valAlphMeterC = utils.AverageMeter()
                        valAlphMeterR = utils.AverageMeter()

                        for xy_val in valid_loader:
                            xy_val = cvt(xy_val)
                            nex = xy_val.shape[0]
                            x_val = xy_val[:, :100].view(-1, 100) @ Vx
                            y_val = xy_val[:, 100:].view(-1, dy)

                            val_loss, val_costs = compute_loss(net_x, x_val, net_y(y_val), nt=nt_val)

                            # update average meters
                            valLossMeter.update(val_loss.item(), nex)
                            valAlphMeterL.update(val_costs[0].item(), nex)
                            valAlphMeterC.update(val_costs[1].item(), nex)
                            valAlphMeterR.update(val_costs[2].item(), nex)

                        Loss = valLossMeter.avg
                        Lx = valAlphMeterL.avg
                        Cx = valAlphMeterC.avg
                        Rx = valAlphMeterR.avg

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
                            bestParams_y = net_y.state_dict()
                            # save model
                            torch.save({
                                'args': args,
                                'state_dict_x': bestParams_x,
                                'state_dict_y': bestParams_y,
                            }, os.path.join(args.save, start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' +
                                            f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth'))
                        else:
                            n_vals_wo_improve_netx += 1
                        log_message += 'netx no improve: {:d}/{:d}'.format(n_vals_wo_improve_netx, args.early_stopping)

                    net_x.train()
                    net_y.train()

                if itr % args.viz_freq == 0 or itr == total_itr:
                    net_x.eval()
                    net_y.eval()
                    with torch.no_grad():
                        # get first example from validation data
                        for batch_idx, batch in enumerate(valid_loader):
                            if batch_idx == 0:
                                xy_val = cvt(batch[:8])
                                x_val = xy_val[:, :100].repeat(8, 1) @ Vx
                                y_val = xy_val[:, 100:].repeat(8, 1)
                                break
                        # sample from conditional distribution
                        z = torch.randn_like(x_val)
                        u_val = net_y(y_val)
                        f_inv = integrate(z, u_val, net_x, [1.0, 0.0], nt_val, stepper="rk4", alph=net_x.alph)
                        x_gen = (f_inv[:, :dx] @ Vx.T).detach().cpu()

                    # make Figure with 2x4 subplots
                    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
                    axs = axs.flatten()

                    # loop over the 8 examples
                    for i in range(8):
                        loss_i, costs_i = compute_loss(net_x, x_val[i].view(1, -1), net_y(y_val[i].view(1, -1)), nt=nt_val)
                        axs[i].plot(x_gen[i::8].T, color='gray', linewidth=0.5)
                        axs[i].plot((x_val[i] @ Vx.T).detach().cpu(), color='red', linewidth=2)
                        # set title to loss_i
                        axs[i].set_title(f'NLL: {costs_i[1].item():.2f}')
                        axs[i].set_ylim([-3, 3])
                    # save the figure
                    fig.savefig(os.path.join(args.save, start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' +
                                             f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_cond_samples_itr-{itr}.png'))
                    plt.close()
                    net_x.train()
                    net_y.train()

                # end validation
                logger.info(log_message)  # print iteration

                # stop if NLL is above threshold
                if costs_x[1] > 1e4:
                    flag = 1
                    break

                if args.drop_freq == 0:  # if set to the code setting 0 , the lr drops based on validation
                    if n_vals_wo_improve_netx > args.early_stopping:
                        if ndecs_netx > 2:
                            flag = 2
                            break
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

    if flag == 0:
        logger.info("Training completed")
    elif flag == 1:
        logger.info("NLL is too large...exiting")
    elif flag == 2:
        logger.info("Training stopped early")

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + f'_{args.data}_alph{net_x.alph[0]:.2f}_{net_x.alph[1]:.2f}' +
                f'_{net_x.alph[2]:.2f}_{args.batch_size}_{args.lr}_{m}_{args.nt}_checkpt.pth')
    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
