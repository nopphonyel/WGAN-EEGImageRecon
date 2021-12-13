import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from dataset.fMRI_HC import fMRI_HC_Dataset

from libs.model import SimpleFCExtractor
from libs.model import *
import libs.model.wgans.experimental as wgans_experimental
import libs.model.wgans.sa_model as sa_model
from libs.model.wgans import AlexNetExtractor
from libs.utils.logger import LoggerGroup, Reporter
from libs.utils import *
from libs.loss import j1_loss

import torch.nn as nn

import matplotlib.pyplot as plt

import random

# torch.autograd.set_detect_anomaly(True)

dev = "cuda"
# dev = torch.device(get_freer_gpu()) if torch.cuda.is_available() else torch.device("cpu")
ngpu = 1
EPOCHS = 5000
LR = 1e-4
BS = 16
load_at_epoch = 0
LOAD_GEN = False
LOAD_DIS = False
d_epoch_steps = 5
g_epoch_steps = 1
z_dim = 100
lambda_gp = 10

preview_gen_num = 5
export_gen_img_every = 8

# Define some path variable
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "export_content/saved_models/%s/" % fMRI_HC_Dataset.get_name())
IMAGE_PATH = os.path.join(__dirname__, "export_content/images/%s/" % fMRI_HC_Dataset.get_name())
# Also create a necessary directory to export stuff
mkdir(MODEL_PATH)
mkdir(IMAGE_PATH)

# Set random seeds
manualSeed = 794
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Check device config
print("<I> Device =", dev)

# ----Dataset declaration----
ds = fMRI_HC_Dataset(p_id=1, v=1).to(dev)
ds_val = fMRI_HC_Dataset(p_id=1, v=1, train=False).to(dev)

ld = DataLoader(ds, batch_size=BS, shuffle=True)
ld_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=True)

# ----Model declaration----
# - Classifier section
non_img_model = SimpleFCExtractor(in_features=948, num_layers=4, num_classes=6, latent_idx=2, latent_size=200).to(dev)
# non_img_model = StupidFC01(latent_size=200, num_classes=6).to(dev)
# non_img_model = FCAndMultihead(latent_size=200, num_classes=6, input_size=948, head_num=4)
# non_img_model = BiLSTMMultihead(
#     latent_size=200,
#     num_classes=ds.get_num_classes(),
#     data_len=948,
#     lstm_hidden_size=2
# ).to(dev)
img_model = AlexNetExtractor(output_class_num=6, in_channel=1, feature_size=200, pretrain=False).to(dev)

nimg_optim = torch.optim.Adam(non_img_model.parameters(), lr=LR)
img_optim = torch.optim.Adam(img_model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

# - WGANs section
# netD = wgans.Discriminator(ngpu=1, num_classes=ds.get_num_classes(), img_channel=1).to(dev)
# netG = wgans.Generator(ngpu=1, num_classes=ds.get_num_classes(), img_channel=1).to(dev)
num_class = ds.get_num_classes()
netD = sa_model.SADiscriminator(ngpu=1, num_classes=num_class, img_channel=1, latent_size=200).to(dev)
netG = wgans_experimental.GeneratorNoLabel(ngpu=1, num_classes=num_class, img_channel=1, latent_size=200).to(dev)

if (dev == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    netG = nn.DataParallel(netG, list(range(ngpu)))

# - Init model with some weight or resume from previous training
if load_at_epoch != 0:
    if LOAD_GEN:
        netG.load_state_dict(torch.load(MODEL_PATH + "%d_G.pth" % load_at_epoch))
    if LOAD_DIS:
        netD.load_state_dict(torch.load(MODEL_PATH + "%d_D.pth" % load_at_epoch))
else:
    netD.apply(model_utils.weights_init)
    netG.apply(model_utils.weights_init)

d_optim = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.0, 0.9))
g_optim = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.0, 0.9))

# ----Value tracker declaration----
loss_logger = LoggerGroup("Loss")
acc_logger = LoggerGroup("Accuracy")
wgan_logger = LoggerGroup("WGANs")

loss_logger.add_var('total', 'j1', 'j2', 'j3')
acc_logger.add_var('j2_acc', 'j3_acc')
wgan_logger.add_var('g_loss', 'd_loss')

reporter = Reporter(loss_logger, acc_logger, wgan_logger)

try:
    for e in range(EPOCHS):
        # Train session
        for i, (fmri, img, label_idx) in enumerate(ld):
            curr_bs = fmri.shape[0]

            l_p = F.one_hot(label_idx, num_classes=6).float()
            fy_p, ly_p = non_img_model(fmri)
            fx_p, lx_p = img_model(img)

            j1 = j1_loss(l_p, l_p, fy_p, fx_p)
            j2 = criterion(ly_p, label_idx)
            j3 = criterion(lx_p, label_idx)

            loss = j1 + j2 + j3

            nimg_optim.zero_grad()
            img_optim.zero_grad()
            loss.backward()
            nimg_optim.step()
            img_optim.step()

            # Report
            loss_logger.collect_step('total', loss.item())
            loss_logger.collect_step('j1', j1.item())
            loss_logger.collect_step('j2', j2.item())
            loss_logger.collect_step('j3', j3.item())

            # Train netD
            for _ in range(d_epoch_steps):
                fy_p, ly_p = non_img_model(fmri)
                ly_p_idx = torch.argmax(ly_p, dim=1)

                zz = torch.randn(curr_bs, z_dim, 1, 1, device=dev)
                ld_real = netD(img, fy_p, label_idx).view(-1)
                fake_img = netG(zz, fy_p)
                ld_fake = netD(fake_img, fy_p, label_idx).view(-1)

                gp = model_utils.gradient_penalty(netD, fy_p, label_idx, img, fake_img, dev)
                d_loss = -(torch.mean(ld_real) - torch.mean(ld_fake)) + lambda_gp * gp
                wgan_logger.collect_sub_step('d_loss', d_loss.item())

                d_optim.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optim.step()

            # Train netG
            for _ in range(g_epoch_steps):
                fy_p, ly_p = non_img_model(fmri)
                ly_p_idx = torch.argmax(ly_p, dim=1)

                zz = torch.randn(curr_bs, z_dim, 1, 1, device=dev)
                fake_img = netG(zz, fy_p)
                ld_fake = netD(fake_img, fy_p, label_idx).view(-1)

                g_loss = -torch.mean(ld_fake)

                # Test back prop again... in hope that discriminator also improve the non-img
                g_optim.zero_grad()
                # nimg_optim.zero_grad()
                g_loss.backward(retain_graph=True)
                g_optim.step()
                # nimg_optim.step()

                wgan_logger.collect_sub_step('g_loss', g_loss.item())
            wgan_logger.flush_sub_step_all()

            # Validate stuff
            for fmri_val, img_val, label_idx_val in ld_val:
                non_img_model.eval()
                img_model.eval()

                fy_p, ly_p = non_img_model(fmri_val)
                _, lx_p = img_model(img_val)
                ly_p_idx = torch.argmax(ly_p, dim=1)
                lx_p_idx = torch.argmax(lx_p, dim=1)
                j2_acc = torch.sum(ly_p_idx == label_idx_val) / label_idx_val.shape[0]
                j3_acc = torch.sum(lx_p_idx == label_idx_val) / label_idx_val.shape[0]

                # Only export image when end of 'export_gen_img_every'th epoch
                if i == 0 and (e + 1) % export_gen_img_every == 0:
                    zz = torch.randn(preview_gen_num, z_dim, 1, 1, device=dev)
                    fy_p = fy_p[0:preview_gen_num, :]
                    ly_p_idx = ly_p_idx[0:preview_gen_num]
                    fake_img = netG(zz, fy_p)

                    real_img = make_grid(img_val[0:preview_gen_num, :, :, :], nrow=preview_gen_num, normalize=True)
                    fake_img = make_grid(fake_img, nrow=preview_gen_num, normalize=True)
                    img_grid = torch.cat((real_img, fake_img), 1)
                    save_image(img_grid, IMAGE_PATH + "epoch_{}.png".format(e), normalize=False)

                acc_logger.collect_step('j2_acc', j2_acc.item() * 100)
                acc_logger.collect_step('j3_acc', j3_acc.item() * 100)

            non_img_model.train()
            img_model.train()

            reporter.report(epch=e + 1, b_i=i, b_all=len(ld) - 1)

        loss_logger.flush_step_all()
        acc_logger.flush_step_all()
        wgan_logger.flush_step_all()

    reporter.stop()
    loss_logger.plot_all()
    acc_logger.plot_all()

except KeyboardInterrupt:
    reporter.stop()

finally:
    reporter.stop()
