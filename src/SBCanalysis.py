# from https://github.com/mackelab/gatsbi/blob/main/gatsbi/task_utils/shallow_water_model/sbc_analysis.py
from os.path import join
import numpy as np
from src.OTFlowProblem import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_data = ".../OT-Flow/datasets/shallow_water_data3500.npz"


def process_test_data(obs, mean, std, proj_mat, x_dim):
    # project observation
    x_star_proj = proj_mat.T @ obs.cpu().numpy()
    # normalize
    x_star_proj_norm = (x_star_proj.T - mean[:, x_dim:]) / std[:, x_dim:]
    return x_star_proj_norm


def generate_theta(check_point, proj_x, net, embedding, x_cond, mean, std, num_samples, x_dim=14):
    # generate
    normSamples = torch.randn(num_samples, x_dim)
    zx = normSamples.to(device)
    x_cond_tensor = torch.tensor(x_cond, dtype=torch.float32).to(device)
    x_cond_tensor = torch.broadcast_to(x_cond_tensor, (num_samples, 3500))
    x_cond_embed = embedding(x_cond_tensor).reshape(num_samples, -1)
    finvx = integrate(zx, x_cond_embed, net, [1.0, 0.0], check_point['args'].nt_val, stepper="rk4", alph=net.alph)
    modelGen = (finvx[:, :x_dim] @ proj_x.T).detach().cpu().numpy()
    theta_gen = (modelGen * std[:, :100] + mean[:, :100] + 10.0).squeeze()
    return theta_gen


def get_rank_statistic(
    checkpoint,
    Vx,
    net: nn.Module,
    embedding: nn.Module,
    trn_mean,
    trn_std,
    path_to_samples: str,
    num_samples: int = 1000,
    save: bool = False,
    save_dir: str = None,
):
    """
    Calculate rank statistics.

    generator: trained GATSBI generator network.
    path_to_samples: file from which to load groundtruth samples.
    num_samples: number test samples per conditioning variable.
    save: if True, save ranks as npz file.
    save_dir: location at which to save ranks.
    """
    net.to(device)
    embedding.to(device)
    sbc = np.load(path_to_samples)
    thos = torch.FloatTensor(sbc["depth_profile"])
    xos = torch.FloatTensor(sbc["z_vals"])[:, :, :, 1:, :]
    Vs = np.load(path_to_data)["V"]

    # Calculate ranks
    ndim = thos.shape[-1]
    ranks = [[] for _ in range(ndim)]

    f = torch.distributions.Normal(loc=torch.zeros(1), scale=10)
    all_samples = []
    for k, (tho, xo) in enumerate(zip(thos.squeeze(), xos.squeeze())):
        xo_processed = process_test_data(xo.reshape(-1, 1), trn_mean, trn_std, Vs, ndim)
        samples = generate_theta(checkpoint, Vx, net, embedding, xo_processed, trn_mean, trn_std, num_samples)
        samples = torch.FloatTensor(samples)
        all_samples.append(samples.unsqueeze(0))
        # Calculate rank under Gaussian.
        for i in range(ndim):
            slp = f.log_prob(samples[:, i])
            gtlp = f.log_prob(tho[i]+10.0)
            rr = (slp < gtlp).sum().item()
            ranks[i].append(rr)
    all_samples = np.concatenate(all_samples, 0)
    if save:
        np.savez(join(save_dir, "COT_SBC.npz"), ranks=ranks, samples=all_samples)
    return np.array(ranks), all_samples
