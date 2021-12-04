import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.fMRI_HC import fMRI_HC_Dataset

from libs.model import SimpleFCExtractor
from libs.model import *
from libs.model.wgans import AlexNetExtractor
from libs.utils.logger import LoggerGroup, Reporter
from libs.metrices import micro_macro_f1
from libs.loss import j1_loss
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

# torch.autograd.set_detect_anomaly(True)

dev = "cuda"
EPOCHS = 100
LR = 1e-4
BS = 16

# Dataset declaration
ds = fMRI_HC_Dataset(p_id=1, v=1).to(dev)
ds_val = fMRI_HC_Dataset(p_id=1, v=1, train=False).to(dev)

ld = DataLoader(ds, batch_size=BS, shuffle=True)
ld_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=True)

non_img_model = LSTMandAttention(data_len=948, num_classes=6, latent_size=200).to(dev)
# non_img_model = SimpleFCExtractor(in_features=948, num_layers=4, num_classes=6, latent_idx=2, latent_size=200).to(dev)
# non_img_model = StupidFC01(latent_size=200, num_classes=6)
# non_img_model = FCAndMultihead(latent_size=200, num_classes=6, input_size=948, head_num=4)
# non_img_model = BiLSTMMultihead2xEncoder(latent_size=200, num_classes=ds.get_num_classes(), data_len=948, lstm_hidden_size=2)
# non_img_model = BiLSTMMultihead(latent_size=200, num_classes=ds.get_num_classes(), data_len=948, lstm_hidden_size=2)
# non_img_model = DumbAssFC(latent_size=200, num_classes=6)
# non_img_model = Conv1DStuff(latent_size=200, num_classes=6, data_len=948)
img_model = AlexNetExtractor(output_class_num=6, in_channel=1, feature_size=200, pretrain=False).to(dev)

nimg_optim = torch.optim.Adam(non_img_model.parameters(), lr=LR)
img_optim = torch.optim.Adam(img_model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

loss_logger = LoggerGroup("Loss")
acc_logger = LoggerGroup("Accuracy")
div_logger = LoggerGroup("Divergence")
f1_logger = LoggerGroup("F1 Non-img score")

loss_logger.add_var('total', 'j1', 'j2', 'j3')
acc_logger.add_var('j2_acc', 'j3_acc')
div_logger.add_var('kl_x2y', 'kl_y2x', 'jl')
f1_logger.add_var('micro', 'macro')

reporter = Reporter(loss_logger, acc_logger, div_logger, f1_logger)
reporter.classic_reporter(True)

for e in range(EPOCHS):
    # Train session
    for i, (fmri, img, label_idx) in enumerate(ld):
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

        kl_y2x = torch.nn.functional.kl_div(fy_p, fx_p, reduction='batchmean')
        kl_x2y = torch.nn.functional.kl_div(fx_p, fy_p, reduction='batchmean')
        jl = (kl_x2y * 0.5) + (kl_y2x * 0.5)

        div_logger.collect_step('kl_y2x', kl_y2x.item())
        div_logger.collect_step('kl_x2y', kl_x2y.item())
        div_logger.collect_step('jl', jl.item())

        # Validate stuff
        for fmri_val, img_val, label_idx_val in ld_val:
            non_img_model.eval()
            img_model.eval()
            _, ly_p = non_img_model(fmri_val)
            _, lx_p = img_model(img_val)
            ly_p_idx = torch.argmax(ly_p, dim=1)
            lx_p_idx = torch.argmax(lx_p, dim=1)
            j2_acc = (torch.sum(ly_p_idx == label_idx_val) / label_idx_val.shape[0]).item()
            j3_acc = (torch.sum(lx_p_idx == label_idx_val) / label_idx_val.shape[0]).item()
            acc_logger.collect_step('j2_acc', j2_acc * 100)
            acc_logger.collect_step('j3_acc', j3_acc * 100)

            micro, macro = micro_macro_f1(pred=ly_p_idx, real=label_idx_val, num_classes=6)
            f1_logger.collect_step('micro', micro)
            f1_logger.collect_step('macro', macro)

        non_img_model.train()
        img_model.train()

        reporter.report(epch=e + 1, b_i=i, b_all=len(ld) - 1)

    loss_logger.flush_step_all()
    acc_logger.flush_step_all()
    f1_logger.flush_step_all()

loss_logger.plot_all()
acc_logger.plot_all()
f1_logger.plot_all()

# plt.title("Full history")
# plt.plot(j1_fhist, label="j1")
# plt.plot(j2_fhist, label="j2")
# plt.plot(j3_fhist, label="j3")
# plt.plot(loss_fhist, label="loss")
# plt.legend()
# plt.show()
#
# plt.title("Epoch history")
# plt.plot(j1_hist, label="j1")
# plt.plot(j2_hist, label="j2")
# plt.plot(j3_hist, label="j3")
# plt.plot(loss_hist, label="loss")
# plt.legend()
# plt.show()
