import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from dataset.EMNIST_Extended import EMNIST_Extended
from dataset.fMRI_HC import fMRI_HC_Dataset

import libs.model.wgans.experimental as wgans_experimental
from libs.metrices import micro_macro_f1
from libs.model import *
from libs.model.fe.experimental import AlexNetEncoder, SimpleFCEncoder
from libs.utils.logger import LoggerGroup, Reporter
from libs.utils import *
from libs.loss import j1_loss

import torch.nn as nn

import random

# torch.autograd.set_detect_anomaly(True)

dev = "cuda"
# dev = torch.device(get_freer_gpu()) if torch.cuda.is_available() else torch.device("cpu")
ngpu = 1
EPOCHS = 10000
LR = 1e-4
BS = 16
load_at_epoch = 0
LOAD_GEN = False
LOAD_DIS = False
d_epoch_steps = 5
g_epoch_steps = 1
z_dim = 100
lambda_gp = 10

preview_gen_num = 6
export_gen_img_every = 20

# Define some path variable
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "export_content/saved_models/%s/" % fMRI_HC_Dataset.get_name())
IMAGE_PATH = os.path.join(__dirname__, "export_content/images/%s/" % fMRI_HC_Dataset.get_name())
LOGGER_PATH = os.path.join(__dirname__, "export_content/log/")

# Also create a necessary directory to export stuff
mkdir(MODEL_PATH)
mkdir(IMAGE_PATH)
mkdir(LOGGER_PATH)

# Set random seeds
manualSeed = 794
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Check device config
print("<I> Device =", dev)

# ----Dataset declaration----
ds = fMRI_HC_Dataset(p_id=1, v=1).to(dev)
ds_val = fMRI_HC_Dataset(p_id=1, v=1, train=False).to(dev)
ds_u = EMNIST_Extended(train=True).to(dev)
ds_u_val = EMNIST_Extended(train=False).to(dev)

ld = DataLoader(ds, batch_size=BS, shuffle=True)
ld_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=True)
ld_u = DataLoader(ds_u, batch_size=BS * 4, shuffle=True)
ld_u_val = DataLoader(ds_u, batch_size=len(ds_val), shuffle=True)
# ld_u = DataLoader(ds, batch_size=BS*8, shuffle=True)

# ----Model declaration----
# - Classifier section
# non_img_model = SimpleFCExtractor(in_features=948, num_layers=4, num_classes=6, latent_idx=2, latent_size=200).to(dev)
non_img_model = SimpleFCEncoder(in_features=948, num_layers=4, latent_size=200).to(dev)
# non_img_model = StupidFC01(latent_size=200, num_classes=6).to(dev)
# non_img_model = FCAndMultihead(latent_size=200, num_classes=6, input_size=948, head_num=4)
# non_img_model = BiLSTMMultihead(
#     latent_size=200,
#     num_classes=ds.get_num_classes(),
#     data_len=948,
#     lstm_hidden_size=2
# ).to(dev)
# img_model = AlexNetExtractor(output_class_num=6, in_channel=1, feature_size=200, pretrain=False).to(dev)
img_model = AlexNetEncoder(in_channel=1, feature_size=200, pretrain=False).to(dev)

nimg_optim = torch.optim.Adam(non_img_model.parameters(), lr=LR)
img_optim = torch.optim.Adam(img_model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

# - WGANs section
# netD = wgans.Discriminator(ngpu=1, num_classes=ds.get_num_classes(), img_channel=1).to(dev)
# netG = wgans.Generator(ngpu=1, num_classes=ds.get_num_classes(), img_channel=1).to(dev)
# num_class = ds.get_num_classes()
netD = wgans_experimental.DiscriminatorNoLabel(ngpu=1, img_channel=1, latent_size=200).to(dev)
netG = wgans_experimental.GeneratorNoLabel(ngpu=1, img_channel=1, latent_size=200).to(dev)

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
# loss_logger = LoggerGroup("Loss")
# acc_logger = LoggerGroup("Accuracy")  # There will be no classification accuracy  # TODO: Check later

wgan_logger = LoggerGroup("WGANs")
# f1_logger = LoggerGroup("F1 Score")

# loss_logger.add_var('total', 'j1', 'j2', 'j3', 'j4', 'j5')

wgan_logger.add_var('g_loss', 'd_loss')
# f1_logger.add_var('micro', 'macro')

reporter = Reporter(wgan_logger)

for e in range(EPOCHS):
    # Train session
    for i, ((fmri, img, _), (img_u, _)) in enumerate(zip(ld, ld_u)):
        curr_bs = fmri.shape[0]
        curr_bs_u = img_u.shape[0]

        # l_p = F.one_hot(label_idx, num_classes=6).float()
        # fy_p = non_img_model(fmri)
        # fx_p = img_model(img)

        # l_u = F.one_hot(label_idx_u, num_classes=6).float()
        # fx_u = img_model(img_u)

        # j1 = j1_loss(l_p, l_p, fy_p, fx_p)
        # j2 = criterion(ly_p, label_idx)
        # j3 = criterion(lx_p, label_idx)
        # j4 = j1_loss(l_p, l_u, fx_u, fy_p)
        # j5 = criterion(lx_u, label_idx_u)

        # loss = j1 + j2 + j3 + j4 + j5
        #
        # nimg_optim.zero_grad()
        # img_optim.zero_grad()
        # loss.backward()
        # nimg_optim.step()
        # img_optim.step()

        # Report
        # loss_logger.collect_step('total', loss.item())
        # loss_logger.collect_step('j1', j1.item())
        # loss_logger.collect_step('j2', j2.item())
        # loss_logger.collect_step('j3', j3.item())
        # loss_logger.collect_step('j4', j4.item())
        # loss_logger.collect_step('j5', j5.item())

        # Train netD
        for _ in range(d_epoch_steps):
            fy_p = non_img_model(fmri)
            fx_u = img_model(img_u)

            # !!! From here, I will try with concat paired and unpaired latent together first
            # Thus : img + img_u, fy + fx, ly_p_idx + lx_u_idx needs to concat together
            zz = torch.randn(curr_bs + curr_bs_u, z_dim, 1, 1, device=dev)
            img_pack = torch.cat((img, img_u), dim=0)
            f_pack = torch.cat((fy_p, fx_u), dim=0)
            # l_pack_idx = torch.cat((ly_p_idx, lx_u_idx), dim=0)  # TODO : Check discrim training stuff

            # ld_real = netD(img, fy_p, ly_p_idx).view(-1)
            # fake_img = netG(zz, ly_p_idx, fy_p)
            # ld_fake = netD(fake_img, fy_p, ly_p_idx).view(-1)
            ld_real = netD(img_pack, f_pack).view(-1)
            fake_img = netG(zz, f_pack)
            ld_fake = netD(fake_img, f_pack).view(-1)

            # gp = model_utils.gradient_penalty(netD, fy_p, ly_p_idx, img, fake_img, dev)
            gp = model_utils.gradient_penalty_no_label(netD, f_pack, img_pack, fake_img, dev)
            d_loss = -(torch.mean(ld_real) - torch.mean(ld_fake)) + lambda_gp * gp
            wgan_logger.collect_sub_step('d_loss', d_loss.item())

            d_optim.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optim.step()

        # Train netG
        for _ in range(g_epoch_steps):
            fy_p = non_img_model(fmri)
            fx_u = img_model(img_u)

            # ly_p_idx = torch.argmax(ly_p, dim=1)
            # lx_u_idx = torch.argmax(lx_u, dim=1)

            # !!! Let's concat stuff from here
            zz = torch.randn(curr_bs + curr_bs_u, z_dim, 1, 1, device=dev)
            # l_pack_idx = torch.cat((ly_p_idx, lx_u_idx), dim=0)
            f_pack = torch.cat((fy_p, fx_u), dim=0)

            fake_img = netG(zz, f_pack)
            ld_fake = netD(fake_img, f_pack).view(-1)

            g_loss = -torch.mean(ld_fake)

            # Test back prop again... in hope that discriminator also improve the non-img
            g_optim.zero_grad()
            nimg_optim.zero_grad()
            img_optim.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optim.step()
            nimg_optim.step()
            img_optim.step()

            wgan_logger.collect_sub_step('g_loss', g_loss.item())
        wgan_logger.flush_sub_step_all()

        # Validate stuff
        for (fmri_val, img_val, _), (img_u_val, _) in zip(ld_val, ld_u_val):
            non_img_model.eval()
            img_model.eval()

            # Concat paired and unpaired together
            img_pack = torch.cat((img_val, img_u_val), dim=0)
            # label_idx_pack = torch.cat((label_idx_val, label_idx_u_val), dim=0)

            # Classifier validation section
            fy_p = non_img_model(fmri_val)
            # _, lx_p_pack = img_model(img_pack)

            # ly_p_idx = torch.argmax(ly_p, dim=1)
            # lx_p_pack_idx = torch.argmax(lx_p_pack, dim=1)
            # j2_acc = torch.sum(ly_p_idx == label_idx_val) / label_idx_val.shape[0]
            # j3_acc = torch.sum(lx_p_pack_idx == label_idx_pack) / label_idx_pack.shape[0]
            # j2_acc = j2_acc.item()
            # j3_acc = j3_acc.item()

            # WGAN validation section
            # Only export image when end of 'export_gen_img_every'th epoch
            if i == 0 and (e + 1) % export_gen_img_every == 0:
                zz = torch.randn(img_val.shape[0], z_dim, 1, 1, device=dev)
                fake_img = netG(zz, fy_p)

                # f1 calculation section
                # _, lx_p = img_model(fake_img)
                # lx_p_idx = torch.argmax(lx_p, dim=1)
                # micro, macro = micro_macro_f1(pred=lx_p_idx, real=label_idx_val, num_classes=ds.get_num_classes())
                # f1_logger.collect_step('micro', micro)
                # f1_logger.collect_step('macro', macro)

                # Image generation export preview
                real_img_grd = make_grid(img_val[0:preview_gen_num, :, :, :], nrow=preview_gen_num, normalize=True)
                fake_img_grd = make_grid(fake_img[0:preview_gen_num, :, :, :], nrow=preview_gen_num, normalize=True)
                img_grid = torch.cat((real_img_grd, fake_img_grd), 1)
                save_image(img_grid, IMAGE_PATH + "epoch_{}.png".format(e), normalize=False)

        non_img_model.train()
        img_model.train()

        reporter.report(epch=e + 1, b_i=i, b_all=len(ld) - 1)

    # loss_logger.flush_step_all()

    wgan_logger.flush_step_all()
    # f1_logger.flush_step_all()

# loss_logger.plot_all()
# f1_logger.plot_all()

# loss_logger.export_file(os.path.join(LOGGER_PATH, "loss.lggr"))
wgan_logger.export_file(os.path.join(LOGGER_PATH, "wgan.lggr"))
# f1_logger.export_file(os.path.join(LOGGER_PATH, "f1.lggr"))

reporter.stop()
