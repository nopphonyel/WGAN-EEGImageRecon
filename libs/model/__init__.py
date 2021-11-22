import libs.model.wgans as classic
import libs.model.model_utils as utils

import torch.nn as nn
from math import ceil


class SimpleFCExtractor(nn.Module):
    def __init__(self, in_features, num_layers, latent_idx, latent_size, num_classes):
        super(SimpleFCExtractor, self).__init__()
        self.module_list = nn.ModuleList()
        self.latent_idx = latent_idx
        self.latent_in = 0
        self.latent_out = latent_size

        rdf = (in_features - num_classes) / float(num_layers)
        in_l = in_features
        out_l1 = in_l - rdf

        for i in range(num_layers):
            is_final_layer = (i == num_layers - 1)
            if is_final_layer:
                out_l1 = num_classes
                act = None
            elif i == latent_idx:
                out_l1 = ceil(out_l1)
                self.latent_in = out_l1
            else:
                out_l1 = ceil(out_l1)
                act = 'lrelu'

            m = SimpleFCExtractor.__create_block(
                in_f=int(in_l),
                out_f=out_l1,
                act=act,
                final_layer=is_final_layer
            )
            # print("{} fc {} -> {}".format(i, in_l, out_l1))
            self.module_list.append(m)
            in_l = out_l1
            out_l1 = in_l - rdf
        self.l_latent_out = SimpleFCExtractor.__create_block(in_f=self.latent_in, out_f=self.latent_out,
                                                             final_layer=True, act=None)

    @staticmethod
    def __create_block(
            in_f: int,
            out_f: int,
            act: str or None,
            dropout_p: float = 0.2,
            final_layer: bool = False
    ):
        """
        This function will create a simple fc block with some appropriate layers
        :param in_f: Number of in features
        :param out_f: Number of out features
        :param act: Activation function name
        :param dropout_p:
        :param final_layer:
        :return:
        """
        block = nn.Sequential()
        block.add_module("fc", nn.Linear(in_f, out_f))

        if not final_layer:
            block.add_module('dropout', nn.Dropout(p=dropout_p))

        if act == 'relu':
            block.add_module('act', nn.ReLU())
        elif act == 'softmax':
            block.add_module('act', nn.Softmax())
        elif act == 'lrelu':
            block.add_module('act', nn.LeakyReLU())
        return block

    def forward(self, x):
        latent = None
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
            if i == self.latent_idx:
                latent = x
        latent = self.l_latent_out(latent)
        return latent, x
