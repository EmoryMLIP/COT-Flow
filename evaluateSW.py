import argparse
import time
import shutil
from importlib import import_module
import subprocess
import os
from os import makedirs
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from src.OTFlowProblem import *
from src.SBCanalysis import get_rank_statistic
from scipy.stats import binom
from shallow_water_model.simulator import ShallowWaterSimulator as Simulator
from shallow_water_model.prior import DepthProfilePrior as Prior
import config

cf = config.getconfig()

parser = argparse.ArgumentParser('COT-Flow')
parser.add_argument('--resume', type=str, default="experiments/...")
parser.add_argument('--resume50k', type=str, default="experiments/...")
parser.add_argument('--resume20k', type=str, default="experiments/...")
parser.add_argument('--prec', type=str, default='single', choices=['None', 'single', 'double'],
                    help="overwrite trained precision")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--long_version', action='store_true')
# default is: args.long_version=False , passing  --long_version will take a long time to run to get values for paper
args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

# function for converting tensor to prec and to device
cvt = lambda x: x.type(checkpt['state_dict_x']['A'].dtype).to(device, non_blocking=True)


def _fwd_pass_fourier(profile, seedz):
    _, z = Simulator(outdir=0, fourier=True)(
        profile, seeds_u=[42], seeds_z=[seedz]
    )
    return z


def wave_wout_noise(theta):
    # abs path to solver
    path_to_fcode = '.../COT-Flow/shallow_water_model/shallow_water01_modified.f90'
    # load solver
    try:
        sw = import_module("shallow_water")
    except ModuleNotFoundError:
        bashcommand = "python -m numpy.f2py -c %s -m shallow_water" % path_to_fcode
        subprocess.call(bashcommand.split(" "))
        sw = import_module("shallow_water")
    # set up temporary dir and file
    outdir = int((time.time() % 1) * 1e7)
    makedirs("%07d" % outdir, exist_ok=True)
    file_z = join("%07d" % outdir, "z%s.dat")
    # simulate wave
    sw.shallow_water(theta, int(outdir))
    # read z output into single array
    z = np.zeros((101, 100))
    for i in range(0, 101):
        str_i = "{0:03d}".format(i)
        with open(file_z % str_i, "r") as f:
            z[i] = np.loadtxt(f)

    # Remove save directory to free memory
    shutil.rmtree("%07d" % outdir)
    return z[1:, :]


def process_test_data(obs, proj, mean, std, x_dim=100):
    # project observation
    x_star_proj = proj.T @ obs
    # normalize
    x_star_proj_norm = (x_star_proj.T - mean[:, x_dim:]) / std[:, x_dim:]
    return x_star_proj_norm


def generate_theta(check_point, proj_x, net, embedding, x_cond, mean, std, x_dim=14, num_samples=100):
    # generate
    normSamples = torch.randn(num_samples, x_dim)
    zx = cvt(normSamples)
    x_cond_tensor = cvt(torch.tensor(x_cond, dtype=torch.float32))
    x_cond_tensor = torch.broadcast_to(x_cond_tensor, (num_samples, 3500))
    # start sampling timer
    start = time.time()
    x_cond_embed = embedding(x_cond_tensor).reshape(num_samples, -1)
    finvx = integrate(zx, x_cond_embed, net, [1.0, 0.0], check_point['args'].nt_val, stepper="rk4", alph=net.alph)
    # end timer
    sample_time = time.time() - start
    print("Sampling Time for " + str(check_point['args'].nt_val) + " is: " + str(sample_time))
    print("finvs shape is: " + str(finvx.shape))
    modelGen = (finvx[:, :x_dim] @ proj_x.T).detach().cpu().numpy()
    theta_gen = (modelGen * std[:, :100] + mean[:, :100] + 10.0).squeeze()
    return theta_gen


def plot_post_predict(axis, t, x_cond_wonoise, theta, color, y_lab=True, num_samples=50):
    x_axs = np.linspace(1, 100, 100)
    # plot ground truth at time t
    axis.plot(x_axs, x_cond_wonoise[t, :], c='k')
    # plot posterior predictives using num_samples random samples
    for _ in range(num_samples):
        rand_sample = np.random.randint(low=0, high=theta.shape[0], size=(1,))[0]
        theta_i = theta[rand_sample, :]
        theta_i = np.expand_dims(theta_i, 0)
        # run forward model
        sim = wave_wout_noise(theta_i)
        # plot simulated wave at time t
        axis.plot(x_axs, sim[t, :], c=color, lw=0.2)
    axis.set_xticks([])
    axis.tick_params(axis='y', which='major', labelsize=24)
    if y_lab is True:
        axis.set_ylabel("Amplitude", rotation=90, fontsize=45)


def build_cotflow(check_point):
    # reload model
    m = check_point['args'].m
    alph = check_point['args'].alph
    nTh = check_point['args'].nTh
    m_y = check_point['args'].m_y
    mout_y = check_point['args'].mout_y
    dx = check_point['args'].dx

    argPrec = check_point['state_dict_x']['A'].dtype
    net_x = Phi(nTh=nTh, m=m, dx=dx, dy=mout_y, alph=alph)
    net_y = nn.Sequential(
        nn.Linear(3500, m_y),
        nn.Tanh(),
        nn.Linear(m_y, m_y),
        nn.Tanh(),
        nn.Linear(m_y, mout_y)
    )
    net_x = net_x.to(argPrec)
    net_y = net_y.to(argPrec)
    net_x.load_state_dict(check_point["state_dict_x"])
    net_y.load_state_dict(check_point["state_dict_y"])
    net_x.to(device)
    net_y.to(device)
    return net_x, net_y


def load_data_info(file_path, valid_ratio):
    data = np.load(file_path)['dataset']
    V = np.load(file_path)['Vs']
    trn, vld = train_test_split(data, test_size=valid_ratio, random_state=42)
    mean = np.mean(trn, axis=0, keepdims=True)
    std = np.std(trn, axis=0, keepdims=True)
    return trn, V, mean, std


def compute_loss(net, x, y, nt):
    Jc, cs = OTFlowProblem(x, y, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    color_list = ['r', 'b', 'salmon']
    time_list = [21, 68, 93]

    """Build COT-Flow"""
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    checkpt_50k = torch.load(args.resume50k, map_location=lambda storage, loc: storage)
    checkpt_20k = torch.load(args.resume20k, map_location=lambda storage, loc: storage)
    net_x, net_y = build_cotflow(checkpt)
    net_x_50k, net_y_50k = build_cotflow(checkpt_50k)
    net_x_20k, net_y_20k = build_cotflow(checkpt_20k)

    """Grab Training Mean and STD"""

    # TODO change to correct path
    file_path = '.../COT-Flow/datasets/shallow_water_data3500.npz'
    file_path50k = '.../COT-Flow/datasets/shallow_water_data3500_50k.npz'
    file_path20k = '.../COT-Flow/datasets/shallow_water_data3500_20k.npz'

    train_data, Vs, train_mean, train_std = load_data_info(file_path, 0.05)
    train_data_50k, Vs50k, train_mean_50k, train_std_50k = load_data_info(file_path50k, 0.05)
    train_data_20k, Vs20k, train_mean_20k, train_std_20k = load_data_info(file_path20k, 0.05)

    dx = checkpt['args'].dx
    if dx < 100:
        x_full = torch.FloatTensor(train_data[:, :100]).view(-1, 100)
        x_full_50k = torch.FloatTensor(train_data_50k[:, :100]).view(-1, 100)
        x_full_20k = torch.FloatTensor(train_data_20k[:, :100]).view(-1, 100)
        cov_x = x_full.T @ x_full
        cov_x_50k = x_full_50k.T @ x_full_50k
        cov_x_20k = x_full_20k.T @ x_full_20k
        L, V = torch.linalg.eigh(cov_x)
        L50k, V50k = torch.linalg.eigh(cov_x_50k)
        L20k, V20k = torch.linalg.eigh(cov_x_20k)
        # get the last dx columns in V
        Vx = cvt(V[:, -dx:])
        Vx50k = cvt(V50k[:, -dx:])
        Vx20k = cvt(V20k[:, -dx:])
    else:
        Vx, Vx50k, Vx20k = cvt(torch.eye(100)), cvt(torch.eye(100)), cvt(torch.eye(100))

    """Sample and Plotting"""

    # sample for ground truth prior
    seed_depth = 77777
    theta_star = Prior(return_seed=False)(seed=seed_depth)

    # obtain x_star=f(theta_star)
    x_fourier = _fwd_pass_fourier(theta_star, seedz=seed_depth)
    x_vals_fourier = x_fourier.squeeze()
    x_vals_fourier = x_vals_fourier[:, 1:, :]
    x_star_fourier = x_vals_fourier.reshape(-1, 1)

    # obtain noiseless wave from theta_star
    x_star_nofourier_nonosie = wave_wout_noise(theta_star)

    # generate theta from COT-Flow
    x_star_processed = process_test_data(x_star_fourier, Vs, train_mean, train_std)
    theta_samples = generate_theta(checkpt, Vx, net_x, net_y, x_star_processed, train_mean, train_std)

    """MAP estimation"""

    theta = torch.randn(1, 14, requires_grad=True).to(device)
    theta_min = theta.clone().detach().requires_grad_(True)

    def closure():
        u = net_y(torch.tensor(x_star_processed, dtype=theta.dtype).to(device)).reshape(1, -1)
        _, cost = compute_loss(net_x, theta_min, u, nt=32)
        loss = cost[1]
        theta_min.grad = torch.autograd.grad(loss, theta_min)[0].detach()
        return loss

    optimizer = torch.optim.LBFGS([theta_min], line_search_fn="strong_wolfe", max_iter=1000000)
    optimizer.step(closure)
    theta_map = (theta_min @ Vx.T).detach().cpu().numpy()
    theta_map = (theta_map * train_std[:, :100] + train_mean[:, :100] + 10.0).squeeze()

    """COT Posterior Plotting"""

    # create plot grid for ground truth values
    fig, axs = plt.subplots(1, 5)
    fig.set_size_inches(40, 7)

    # plot posterior samples with ground truth theta
    xx = np.linspace(1, 100, 100)
    axs[0].set_ylim(bottom=4.0, top=18.0)
    axs[0].plot(xx, theta_star.squeeze(0), c='k')
    axs[0].scatter(xx, theta_map, c='m', marker='x', s=256)
    for i in range(theta_samples.shape[0]):
        thetai = theta_samples[i, :]
        axs[0].plot(xx, thetai, c='grey', lw=0.2)
    axs[0].set_xticks([])
    axs[0].tick_params(axis='y', which='major', labelsize=24)
    axs[0].set_ylabel("Depth Profile", rotation=90, fontsize=45)

    # plot 2d inferred wave image
    sim_wave = wave_wout_noise(theta_samples[0, :].reshape(1, -1))
    img_sim = axs[1].imshow(sim_wave, cmap='gray')
    axs[1].axhline(time_list[0], color=color_list[0], linewidth=4)
    axs[1].axhline(time_list[1], color=color_list[1], linewidth=4)
    axs[1].axhline(time_list[2], color=color_list[2], linewidth=4)
    axs[1].set_xticks([])
    axs[1].tick_params(axis='y', which='major', labelsize=24)
    axs[1].margins(0.3)
    axs[1].set_ylabel("Time", rotation=90, fontsize=45)
    axs[1].invert_yaxis()

    # plot posterior predictives with ground truth wave
    time_list = [21, 68, 93]
    # plot at three times
    plot_post_predict(axs[2], time_list[0], x_star_nofourier_nonosie, theta_samples, color=color_list[0])
    plot_post_predict(axs[3], time_list[1], x_star_nofourier_nonosie, theta_samples, color=color_list[1], y_lab=False)
    plot_post_predict(axs[4], time_list[2], x_star_nofourier_nonosie, theta_samples, color=color_list[2], y_lab=False)

    # save
    fig.tight_layout()
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_cot_figure.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Perform SBC Analysis"""

    path_to_samps = '.../COT-Flow/datasets/sw_test_data.npz'
    ranks, _ = get_rank_statistic(checkpt, Vx, net_x, net_y, train_mean, train_std, path_to_samps)

    # plot ranks
    ndim, N = ranks.shape
    nbins = N
    repeats = 1
    hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
    hbb = hb.cumsum() / hb.sum()
    lower = [binom(N, p=p).ppf(0.005) for p in hbb]
    upper = [binom(N, p=p).ppf(0.995) for p in hbb]

    # Plot CDF
    fig = plt.figure(figsize=(8, 5.5))
    fig.tight_layout(pad=3.0)
    spec = fig.add_gridspec(ncols=1, nrows=1)
    ax = fig.add_subplot(spec[0, 0])
    for i in range(ndim):
        hist, *_ = np.histogram(ranks[i], bins=nbins, density=False)
        histcs = hist.cumsum()
        ax.plot(np.linspace(0, nbins, repeats * nbins),
                np.repeat(histcs / histcs.max(), repeats),
                color='b',
                alpha=.1
                )
    ax.plot(np.linspace(0, nbins, repeats * nbins),
            np.repeat(hbb, repeats),
            color="k", lw=2,
            alpha=.8,
            label="uniform CDF")
    ax.fill_between(x=np.linspace(0, nbins, repeats * nbins),
                    y1=np.repeat(lower / np.max(lower), repeats),
                    y2=np.repeat(upper / np.max(lower), repeats),
                    color='k',
                    alpha=.5)
    # Ticks and axes
    ax.set_xticks([0, 500, 1000])
    ax.set_xlim([0, 1000])
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.set_xlabel("Rank", fontsize=20)
    ax.set_yticks([0, .5, 1.])
    ax.set_ylim([0., 1.])
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.set_ylabel("CDF", fontsize=20)
    # Legend
    custom_lines = [Line2D([0], [0], color="k", lw=1.5, linestyle="-"),
                    Line2D([0], [0], color='b', lw=1.5, linestyle="-")]
    ax.legend(custom_lines, ['Uniform CDF', 'COT-Flow'], fontsize=17)

    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_cot_sbc.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Plot COT from Different Data Size"""

    x_star_processed50k = process_test_data(x_star_fourier, Vs50k, train_mean_50k, train_std_50k)
    theta_samples50k = generate_theta(checkpt_50k, Vx50k, net_x_50k, net_y_50k, x_star_processed50k,
                                      train_mean_50k, train_std_50k)
    x_star_processed20k = process_test_data(x_star_fourier, Vs20k, train_mean_20k, train_std_20k)
    theta_samples20k = generate_theta(checkpt_20k, Vx20k, net_x_20k, net_y_20k, x_star_processed20k,
                                      train_mean_20k, train_std_20k)

    # grab mean and std
    mean100k = np.mean(theta_samples, axis=0, keepdims=True).squeeze()
    std100k = np.std(theta_samples, axis=0, keepdims=True).squeeze()
    mean50k = np.mean(theta_samples50k, axis=0, keepdims=True).squeeze()
    std50k = np.std(theta_samples50k, axis=0, keepdims=True).squeeze()
    mean20k = np.mean(theta_samples20k, axis=0, keepdims=True).squeeze()
    std20k = np.std(theta_samples20k, axis=0, keepdims=True).squeeze()

    # calculate normed error and plot
    err_100k = np.linalg.norm(theta_star - mean100k) / np.linalg.norm(theta_star)
    err_50k = np.linalg.norm(theta_star - mean50k) / np.linalg.norm(theta_star)
    err_20k = np.linalg.norm(theta_star - mean20k) / np.linalg.norm(theta_star)

    # plot
    font = {'fontname': 'Times'}
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(28, 8)

    xx = np.linspace(1, 100, 100)
    axs[0].plot(xx, theta_star.squeeze(), c='k', label="Ground Truth")
    axs[0].plot(xx, mean20k, c='orange', label="Posterior Mean 20k")
    axs[0].fill_between(xx, (mean20k - std20k), (mean20k + std20k), color='grey', alpha=0.2)
    axs[0].set_ylabel('Depth', fontsize=26)
    axs[0].text(10, 5, f"rel. error = {err_20k:.2f}", fontsize=20, **font)
    axs[0].legend(fontsize="16")

    axs[1].plot(xx, theta_star.squeeze(), c='k', label="Ground Truth")
    axs[1].plot(xx, mean50k, c='b', label="Posterior Mean 50k")
    axs[1].fill_between(xx, (mean50k - std50k), (mean50k + std50k), color='grey', alpha=0.2)
    axs[1].set_xlabel('Position', fontsize=26)
    axs[1].text(10, 5.35, f"rel. error = {err_50k:.2f}", fontsize=20, **font)
    axs[1].legend(fontsize="16")

    axs[2].plot(xx, theta_star.squeeze(), c='k', label="Ground Truth")
    axs[2].plot(xx, mean100k, c='r', label="Posterior Mean 100k")
    axs[2].fill_between(xx, (mean100k - std100k), (mean100k + std100k), color='grey', alpha=0.2)
    axs[2].text(10, 6.45, f"rel. error = {err_100k:.2f}", fontsize=20, **font)
    axs[2].legend(fontsize="16")

    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_cot_numsims.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, bbox_inches='tight', dpi=300)
    plt.close()
