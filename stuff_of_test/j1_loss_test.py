import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.fMRI_HC import fMRI_HC_Dataset

from libs.model import SimpleFCExtractor
from libs.model.wgans import AlexNetExtractor
from libs.utils.loss_logger import LoggerGroup
from libs.loss import j1_loss
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

dev = "cpu"
EPOCHS = 100
LR = 1e-4
BS = 16

non_img_model = SimpleFCExtractor(in_features=948, num_layers=4, num_classes=6, latent_idx=2, latent_size=200).to(dev)
img_model = AlexNetExtractor(output_class_num=6, in_channel=1, feature_size=200, pretrain=False).to(dev)

nimg_optim = torch.optim.Adam(non_img_model.parameters(), lr=LR)
img_optim = torch.optim.Adam(img_model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

ds = fMRI_HC_Dataset(p_id=1, v=1).to(dev)
ds_val = fMRI_HC_Dataset(p_id=1, v=1, train=False)

ld = DataLoader(ds, batch_size=BS, shuffle=True)
ld_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=True)

loss_logger = LoggerGroup()
acc_logger = LoggerGroup()

loss_logger.add_var('total', 'j1', 'j2', 'j3')
acc_logger.add_var('j2_acc', 'j3_acc')

for e in range(EPOCHS):
    # Train session
    for i, (fmri, img, label_idx) in enumerate(ld):
        l_p = F.one_hot(label_idx, num_classes=6)
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

        # Validate stuff
        for fmri_val, img_val, label_idx_val in ld_val:
            non_img_model.eval()
            img_model.eval()
            _, ly_p = non_img_model(fmri_val)
            _, lx_p = img_model(img_val)
            ly_p_idx = torch.argmax(ly_p, dim=1)
            lx_p_idx = torch.argmax(lx_p, dim=1)
            j2_acc = torch.sum(ly_p_idx == label_idx_val) / label_idx_val.shape[0]
            j3_acc = torch.sum(lx_p_idx == label_idx_val) / label_idx_val.shape[0]
            acc_logger.collect_step('j2_acc', j2_acc * 100)
            acc_logger.collect_step('j3_acc', j3_acc * 100)

        non_img_model.train()
        img_model.train()

        report = "\rEpoch[{:03d} | {}/{}] -> Loss:{:10.4f}, j1[{:10.4f}], j2[{:10.4f}], j3[{:10.4f}] " \
                 "-> Acc: j2[{:7.2f}%], j3[{:7.2f}%]".format(
            e + 1, i,
            len(ld) - 1,
            loss.item(),
            j1.item(),
            j2.item(),
            j3.item(),
            j2_acc.item() * 100,
            j3_acc.item() * 100
        )
        sys.stdout.write(report)

    loss_logger.flush_epoch_all()
    acc_logger.flush_epoch_all()

loss_logger.plot_all()
acc_logger.plot_all()

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
