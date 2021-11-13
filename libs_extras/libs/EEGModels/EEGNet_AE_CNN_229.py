import torch
import torch.nn as nn
import math
from .utils.layers import *


class EEGNet_AE_CNN(nn.Module):
    """
    This model is reimplemented on pytorch followed by this paper : EEGNet: A Compact Convolutional Neural Network
    for EEG-based Brain-Computer Interfaces
    link : https://arxiv.org/abs/1611.08024
    """
    def __init__(self, in_channel, samples, kern_len, F1, F2, D, output_size, dropout_type='2D', dropout_rate=0.5,
                 norm_rate=0.25):
        """
        Expected input shape (BS, 1 (Later will be temporal features), CH, Samples)
        :param in_channel: Number of input EEG channel or electrodes
        :param samples: Number of samples (datapoint in EEG signal on each electrode)
        :param kern_len: (Hyper parameter) The block1 kernel size (In the paper use size = sampling rate//2)
        :param F1: (Hyper parameter) Number of temporal features
        :param F2: (Hyper parameter) Number of pointwise filters
        :param D: (Hyper parameter) Depth multipiler
        :param nb_classes: Number of output class.
        :dropout_type: {'2D', 'classics'} choose your drop out type.
        """
        super(EEGNet_AE_CNN, self).__init__()
        self.is_debug = False
        ################# ENCODER ####################
        self.block1 = nn.Sequential(
            nn.Conv2d(kernel_size=(1, kern_len), in_channels=1, out_channels=F1,
                      padding='same', bias=False),  # Shape=(BS,TEM=F1,CH,LEN=samples)
            nn.BatchNorm2d(num_features=F1),
            ConstraintConv2d(kernel_size=(in_channel, 1), in_channels=F1, groups=F1,
                             out_channels=D * F1, bias=False, weight_max_lim=1.0),
            # Depthwise : shape=(BS, TEM=D*F1, 1,LEN)
            nn.BatchNorm2d(num_features=D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4))
        )

        if dropout_type == '2D':
            self.dropout = nn.Dropout2d(dropout_rate)
        elif dropout_type == 'classics':
            self.dropout = nn.Dropout(dropout_rate)

        # Depthwise : shape=(BS, TEM=D*F1, 1,LEN//4)
        self.block2 = nn.Sequential(
            SeparableConv2d(kernel_size=(1, 16), in_channels=D * F1, out_channels=F2, padding='same', bias=False),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8))
        )

        self.latent_fc = nn.Sequential(
            ConstraintLinear(in_features=F2 * (samples // 32), out_features=512, weight_max_lim=norm_rate),
#             nn.Softmax(dim=1)
        )
        
        ################# DECODER ####################
        self.activation = nn.ELU()
        
        self.dconv1 = nn.Sequential( nn.ConvTranspose2d(in_channels=8,out_channels=6, dilation=1, output_padding=1,  kernel_size=7,stride=2) ,  self.activation) 
        self.dconv2 = nn.Sequential( nn.ConvTranspose2d(in_channels=6,out_channels=5,  dilation=1, output_padding=1,  kernel_size=7,stride=2),   self.activation)
        self.dconv3 = nn.Sequential( nn.ConvTranspose2d(in_channels=5, out_channels=4,  dilation=2, output_padding=1,  kernel_size=7,stride=2),   self.activation)
        self.dconv4 = nn.Sequential( nn.ConvTranspose2d(in_channels=4, out_channels=3,  dilation=1, output_padding=1,  kernel_size=6,stride=2),   self.activation)
        
    
    def forward(self, x):
                                  
        ########## ENCODER ###########
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.latent_fc(x)
                                  
        ########## DECODER ##########    
        x = x.reshape(-1, 8,8,8) 
        x = self.dconv1(x)  
        if self.is_debug : print(x.shape)
        x = self.dconv2(x)
        if self.is_debug : print(x.shape)
        x = self.dconv3(x)
        if self.is_debug : print(x.shape)
        x = self.dconv4(x)
        if self.is_debug : print(x.shape)

#         x = x.reshape(-1, 3, int(math.sqrt(self.output_size//3)), int(math.sqrt(self.output_size//3)))
                                  
        return x

