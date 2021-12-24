from preimport_module import *
from math import ceil


class AlexNetEncoder(nn.Module):
    """
    This class expected image as input with size (64x64x3)
    """

    def __init__(self, in_channel=3, feature_size=200, pretrain=False):
        super(AlexNetEncoder, self).__init__()
        self.feature_size = feature_size
        if not pretrain:
            self.in_channel = in_channel
        else:
            print("<I> Pre-trained has been set, using in_channel=3")
            self.in_channel = 3
        self.features = nn.Sequential(
            # Alex1
            nn.Conv2d(self.in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Alex2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Alex3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # Alex4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Alex5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # return the same number of features but change width and height of img
        if (pretrain):
            import torchvision
            ori_alex = torchvision.models.alexnet(pretrained=True)
            ori_weight = ori_alex.state_dict()
            ori_weight.pop('classifier.1.weight')
            ori_weight.pop('classifier.1.bias')
            ori_weight.pop('classifier.4.weight')
            ori_weight.pop('classifier.4.bias')
            ori_weight.pop('classifier.6.weight')
            ori_weight.pop('classifier.6.bias')
            self.load_state_dict(ori_weight)
            del (ori_alex)
            del (ori_weight)

        self._add_classifier(self.feature_size)

    def _add_classifier(self, feature_size):
        self.fc06 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )
        self.fc07 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc06(x)
        semantic_features = self.fc07(x)
        return semantic_features


class SimpleFCEncoder(nn.Module):
    def __init__(self, in_features, num_layers, latent_size):
        super(SimpleFCEncoder, self).__init__()
        self.module_list = nn.ModuleList()
        self.latent_in = 0
        self.latent_out = latent_size

        rdf = (in_features - latent_size) / float(num_layers)
        in_l = in_features
        out_l1 = in_l - rdf

        for i in range(num_layers):
            is_final_layer = (i == num_layers - 1)
            if is_final_layer:
                out_l1 = ceil(out_l1)
                self.latent_in = out_l1
            else:
                out_l1 = ceil(out_l1)
                act = 'lrelu'

            m = SimpleFCEncoder.__create_block(
                in_f=int(in_l),
                out_f=out_l1,
                act=act,
                final_layer=is_final_layer
            )
            # print("{} fc {} -> {}".format(i, in_l, out_l1))
            self.module_list.append(m)
            in_l = out_l1
            out_l1 = in_l - rdf
        self.l_latent_out = SimpleFCEncoder.__create_block(in_f=self.latent_in, out_f=self.latent_out,
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
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        return x
