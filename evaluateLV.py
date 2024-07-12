import argparse
import scipy.io
import os
import time
import torch
import numpy as np
import pandas as pd
import scipy.stats as st
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.OTFlowProblem import *
from datasets.StochasticLotkaVolterra import StochasticLotkaVolterra
import config
import lib.utils as utils

cf = config.getconfig()

parser = argparse.ArgumentParser('COT-Flow')
parser.add_argument('--resume', type=str, default="experiments/cnf/tabcond/lv/...")
parser.add_argument('--prec', type=str, default='single', choices=['None', 'single', 'double'], help="overwrite trained precision")
parser.add_argument('--gpu' , type=int, default=0)
parser.add_argument('--long_version'  , action='store_true')
# default is: args.long_version=False , passing  --long_version will take a long time to run to get values for paper
args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


def experiment_lv(LV, abc_dat_path, theta_star, net, trn_mean, trn_std, checkpt):

    """Load ytrue from ABC"""

    abc_sample = pd.read_pickle(abc_dat_path)
    y_theta_star = abc_sample["y_true"]
    theta_star_log = np.log(theta_star)
    y_theta_star_norm = (y_theta_star - trn_mean[:, 4:]) / trn_std[:, 4:]
    y_theta_star_norm_tensor = torch.tensor(y_theta_star_norm, dtype=torch.float32)

    """MAP estimation"""

    theta0 = torch.randn(1, 4, requires_grad=True).to(device)
    theta_min = theta0.clone().detach().requires_grad_(True)
    y_cond = y_theta_star_norm_tensor.to(device)

    def closure():
        _, cost = compute_loss(net, theta_min, y_cond, nt=32)
        loss = cost[1]
        theta_min.grad = torch.autograd.grad(loss, theta_min)[0].detach()
        return loss

    optimizer = torch.optim.LBFGS([theta_min], line_search_fn="strong_wolfe", max_iter=1000000)
    optimizer.step(closure)
    theta_map = theta_min.detach().cpu().numpy()
    theta_map = theta_map * trn_std[:, :4] + trn_mean[:, :4]

    """Generate from posterior"""

    y_theta_star_norm_tensor = torch.broadcast_to(y_theta_star_norm_tensor, (2000, 9))
    normSamples = torch.randn(2000, 4).to(device)
    zx = cvt(normSamples)
    # start sampling timer
    start = time.time()
    finvx = integrate(zx, y_theta_star_norm_tensor.to(device), net, [1.0, 0.0], checkpt['args'].nt_val, stepper="rk4",
                      alph=net.alph)
    # end timer
    sample_time = time.time() - start
    print("Sampling Time for " + str(32) + " is: " + str(sample_time))
    modelGen = finvx[:, :4].detach().cpu().numpy()
    theta_gen = (modelGen * trn_std[:, :4] + trn_mean[:, :4]).squeeze()

    """Plot posterior samples and MAP point"""

    symbols = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']
    log_limits = [[-5., 2.], [-5., 2.], [-5., 2.], [-5., 2.]]
    plot_matrix(theta_gen, log_limits, xtrue=theta_star_log, xmap=theta_map.squeeze(), symbols=symbols)
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + str(checkpt['args'].nt_val) +
                         str(theta_star[0].item()) + '.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Plot posterior predictive"""

    plt.figure()
    ytrue = LV.simulate(theta_star)
    c1 = plt.plot(LV.tt, ytrue[:, 0], '-', label='Predators')
    c2 = plt.plot(LV.tt, ytrue[:, 1], '-', label='Prey')
    for i in range(10):
        rand_sample = np.random.randint(low=0, high=2000, size=(1,))[0]
        xi = np.exp(theta_gen[rand_sample, :])
        yt = LV.simulate(xi)
        plt.plot(LV.tt, yt[:, 0], '--', color=c1[0].get_color(), alpha=0.3)
        plt.plot(LV.tt, yt[:, 1], '--', color=c2[0].get_color(), alpha=0.3)
    plt.xlabel('$t$', size=20)
    plt.ylabel('$Z(t)$', size=20)
    plt.legend(loc='upper right')
    plt.xlim(0, 20)
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_' + str(theta_star[0].item()) + '_post.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    # plot for different time steps
    nt_test_list = [16, 8, 4, 2, 1]
    error_array = np.zeros((2, 5))
    error_array[0, :] = nt_test_list
    for dt in nt_test_list:
        zx = cvt(normSamples)
        # start sampling timer
        start = time.time()
        finvx = integrate(zx, y_theta_star_norm_tensor.to(device), net_x, [1.0, 0.0], dt, stepper="rk4", alph=net_x.alph)
        # end timer
        sample_time = time.time() - start
        print("Sampling Time for " + str(dt) + " is: " + str(sample_time))
        modelGen = finvx[:, :4].detach().cpu().numpy()
        theta_gen_new = (modelGen * trn_std[:, :4] + trn_mean[:, :4]).squeeze()
        # grab normed error
        error = np.linalg.norm(theta_gen - theta_gen_new) / np.linalg.norm(theta_gen)
        print(f"Norm Error for theta {theta_star[0].item()} nt = {dt}: {error:.6f}")
        error_array[1, 4-int(np.log2(dt))] = error
        # plots
        plot_matrix(theta_gen_new, log_limits, xtrue=theta_star_log, symbols=symbols)
        sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + str(dt) + str(theta_star[0].item()) + '.png')
        if not os.path.exists(os.path.dirname(sPath)):
            os.makedirs(os.path.dirname(sPath))
        plt.savefig(sPath, dpi=300)
        plt.close()

    # plot error
    plt.figure(figsize=(9, 9))
    plt.scatter(error_array[0, :], error_array[1, :], color='r')
    plt.plot(error_array[0, :], error_array[1, :])
    plt.xlabel("nt", size=20)
    plt.ylabel("error", size=20)
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + str(theta_star[0].item()) + '_err.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plot_matrix(x_samps, limits, xtrue=None, xmap=None, symbols=None):
    dim = x_samps.shape[1]
    plt.figure(figsize=(9, 9))
    for i in range(dim):
        for j in range(i+1):
            ax = plt.subplot(dim, dim, (i*dim)+j+1)
            if i == j:
                plt.hist(x_samps[:, i], bins=40, density=True)
                if xtrue is not None:
                    plt.axvline(xtrue[i], color='r', linewidth=2)
                if xmap is not None:
                    plt.axvline(xmap[i], color='k', linewidth=2)
                plt.xlim(limits[i])
                if i != 0:
                    ax.set_yticklabels([])
                if i != 3:
                    ax.set_xticklabels([])
            else:
                plt.plot(x_samps[:, j], x_samps[:, i], '.k', markersize=.04, alpha=0.1)
                if xtrue is not None:
                    plt.plot(xtrue[j], xtrue[i], '.r', markersize=7, label='Truth')
                if xmap is not None:
                    plt.plot(xmap[j], xmap[i], 'xk', markersize=7, label='Truth')
                if i < 3:
                    ax.set_xticklabels([])
                if j == 1 or j == 2:
                    ax.set_yticklabels([])
                # Peform the kernel density estimate
                xlim = limits[j]   # ax.get_xlim()
                ylim = limits[i]   # ax.get_ylim()
                xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = st.gaussian_kde(x_samps[:, [j, i]].T)
                f = np.reshape(kernel(positions), xx.shape)
                ax.contourf(xx, yy, f, cmap='Blues')
                plt.ylim(limits[i])
            plt.xlim(limits[j])
            if symbols is not None:
                if j == 0:
                    plt.ylabel(symbols[i], size=20)
                if i == len(xtrue)-1:
                    plt.xlabel(symbols[j], size=20)


def compute_loss(net, x, y, nt):
    Jc, cs = OTFlowProblem(x, y, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)

    # reload model
    m = checkpt['args'].m
    alph = checkpt['args'].alph
    nTh = checkpt['args'].nTh

    argPrec = checkpt['state_dict_x']['A'].dtype
    net_x = Phi(nTh=nTh, m=m, dx=4, dy=9, alph=alph)
    net_x = net_x.to(argPrec)
    net_x.load_state_dict(checkpt["state_dict_x"])
    net_x = net_x.to(device)

    # if specified precision supplied, override the loaded precision
    if args.prec != 'None':
        if args.prec == 'single':
            argPrec = torch.float32
        if args.prec == 'double':
            argPrec = torch.float64
        net_x = net_x.to(argPrec)

    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    # load training mean and standard deviation
    # TODO change to correct path
    dataset_load = scipy.io.loadmat('.../COT-Flow/datasets/lv_data.mat')
    x_train = dataset_load['x_train']
    y_train = dataset_load['y_train']
    dataset = np.concatenate((x_train, y_train), axis=1)
    # log transformation over theta
    dataset[:, :4] = np.log(dataset[:, :4])
    # split data and convert to tensor
    train, valid = train_test_split(
        dataset, test_size=0.1,
        random_state=42
    )
    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)

    """Test Generated Sample"""

    # TODO change to correct path
    StochLV = StochasticLotkaVolterra()
    path_theta1 = '.../COT-Flow/datasets/StochasticLV_ABCsamples01.pk'
    theta1 = np.array([0.01, 0.5, 1, 0.01])
    experiment_lv(StochLV, path_theta1, theta1, net_x, train_mean, train_std, checkpt)

    path_theta2 = '.../COT-Flow/datasets/StochasticLV_ABCsamples015NewTheta.pk'
    theta2 = np.array([0.02, 0.02, 0.02, 0.02])
    experiment_lv(StochLV, path_theta2, theta2, net_x, train_mean, train_std, checkpt)

    """Density Estimation"""

    # TODO change to correct path
    test_dataset_load = scipy.io.loadmat('.../COT-Flow/datasets/lv_test_data.mat')
    test_dat = test_dataset_load['test_data']
    # log transformation over theta
    test_dat[:, :4] = np.log(test_dat[:, :4])
    test_data = (test_dat - train_mean) / train_std
    test_data = torch.tensor(test_data, dtype=torch.float32)
    tst_loader = DataLoader(test_data, batch_size=128, shuffle=True)
    nt_de_list = [32, 16, 8, 4, 2, 1]
    for nt in nt_de_list:
        tstLossMeter = utils.AverageMeter()
        tsttimeMeter = utils.AverageMeter()
        for xy in tst_loader:
            xy = cvt(xy)
            x_test = xy[:, :4].view(-1, 4)
            y_test = xy[:, 4:].view(-1, 9)
            # start timer
            end_tst = time.time()
            _, val_costs_x = compute_loss(net_x, x_test, y_test, nt=checkpt['args'].nt_val)
            # end timer
            tststep_time = time.time() - end_tst
            tsttimeMeter.update(tststep_time)
            tstLossMeter.update(val_costs_x[1].item(), xy.shape[0])
        print("Test NLL for nt=" + str(nt) + " is: " + str(tstLossMeter.avg))
        print("Test Time for nt=" + str(nt) + " is: " + str(tsttimeMeter.sum))
