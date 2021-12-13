import torch
import numpy as np
from scipy.linalg import sqrtm


def cov_4d_classics(x, rowvar: bool = False):
    b = x.shape[0]
    f = x.shape[1] * x.shape[2] * x.shape[3]
    x = x.reshape(b, f)
    return np.cov(x, rowvar=rowvar)


def experimental_cov_4d(x):
    N = x.shape[3]
    m1 = x - x.sum(2, keepdims=1) / N
    y_out = np.einsum('ijk,ilk->ijl', m1, m1) / (N - 1)


def FID(gen_img, target_img):
    g_img = gen_img.clone().detach().cpu().numpy()
    t_img = target_img.clone().detach().cpu().numpy()
    mu_1 = np.mean(g_img, axis=(2, 3))
    mu_2 = np.mean(t_img, axis=(2, 3))

    # sig_1 = np.cov(g_img, rowvar=False)
    # sig_2 = np.cov(t_img, rowvar=False)
    sig_1 = cov_4d_classics(g_img, rowvar=False)
    sig_2 = cov_4d_classics(t_img, rowvar=False)

    ssdiff = np.sum((mu_1 - mu_2) ** 2.0)
    covmean = sqrtm(sig_1.dot(sig_2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sig_1 + sig_2 - 2.0 * covmean)
    return fid


BS = 1000
CH = 1
IMG_SIZE = 64

gen_img = torch.rand(BS, CH, IMG_SIZE, IMG_SIZE).float()
tar_img = torch.rand(BS, CH, IMG_SIZE, IMG_SIZE).float()

print(FID(gen_img, tar_img))
