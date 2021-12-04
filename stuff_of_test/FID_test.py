import torch
import numpy as np

def FID(gen_img, target_img):
    g_img = gen_img.detach().cpu().numpy()
    t_img = target_img.detach().cpu().numpy()
    mu_1 = np.mean(g_img, axis=0)
    mu_2 = np.mean(t_img, axis=0)

    sig_1 = np.cov(g_img, rowvar=False)
    sig_2 = np.cov(t_img, rowvar=False)

    ssdiff = np.sum((mu_1 - mu_2)**2.0)
    covmean = np.sqrt(np.dot(sig_1, sig_2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sig_1 + sig_2 - 2.0 * covmean)
    return fid


BS = 3
CH = 1
IMG_SIZE = 64

gen_img = torch.tensor(torch.rand(BS, CH, IMG_SIZE, IMG_SIZE)).float()
tar_img = torch.tensor(torch.rand(BS, CH, IMG_SIZE, IMG_SIZE)).float()

print(FID(gen_img, tar_img))
