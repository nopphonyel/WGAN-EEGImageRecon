import torch
import torch.nn as nn

from .utils.container import ParallelModule
from .utils.layers import ResidualBlock, Flatten
import torch.nn.functional as F


class EEG_ChannelNet(nn.Module):
    """
    This encoder is implemented by follow the architecture from research paper : Decoding Brain Representations 
    by Multimodal Learning of Neural Activity and Visual Features
    link : https://arxiv.org/abs/1810.10974
    """

    def __init__(self, num_class, latent_size=1000):
        """
        Currently, this model limited to a fixed size of EEG Input data in shape of (BS, 1, 128 electrodes, 440 EEG Samples)
        :param output_size (Hyper parameter) The size of output latent vector
        """
        super(EEG_ChannelNet, self).__init__()
        self.temporal_block = ParallelModule(
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 1), padding=(0, 16), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 2), padding=(0, 32), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 4), padding=(0, 64), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 8), padding=(0, 128), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 16), padding=(0, 256), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU())
        )

        self.spatial_block = ParallelModule(
            nn.Sequential(
                nn.Conv2d(kernel_size=(7, 1), stride=(2, 1), dilation=(1, 1), padding=(7, 0), in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(5, 1), stride=(2, 1), dilation=(1, 1), padding=(6, 0), in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(3, 1), stride=(2, 1), dilation=(1, 1), padding=(5, 0), in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 1), stride=(2, 1), dilation=(1, 1), padding=(4, 0), in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU())
        )

        self.residual_block = nn.Sequential(
            ResidualBlock(channel_num=200),
            ResidualBlock(channel_num=200),
            ResidualBlock(channel_num=200),
            ResidualBlock(channel_num=200)
        )
        
        self.output_block = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=200, out_channels=100),
            Flatten(),
            nn.Linear(in_features=2000, out_features=latent_size))
        
        self.classi_fc = nn.Linear(1000, num_class)

    def forward(self, x):
        x = self.temporal_block(x)  # Expected 4D tensor input in shape of : [BS, F, CH, LEN]
#         print("Temporal Block Output : ", x.shape)
        x = self.spatial_block(x)
#         print("Spatial Block Output : ", x.shape)
        x = self.residual_block(x)
#         print("Residual Block Output : ", x.shape)
        x = self.output_block(x)
#         print("output_block Output : ", x.shape)
        return x
    
    def forward_classify(self, x):
        x = self.forward(x)
        x = self.classi_fc(x)
#         x = F.softmax(x, dim=1)
        return x


def channelNetLoss(l_e1: torch.Tensor, l_v1: torch.Tensor, l_v2: torch.Tensor):
    """
    The input should be a latent vector only.
    """
    loss = (l_e1 @ l_v2.transpose(0, 1)) - (l_e1 @ l_v1.transpose(0, 1))
    loss = torch.diag(loss, diagonal=0).reshape(-1, 1).clamp(min=0)
    return loss  # shape should be : [bs, 1] 
