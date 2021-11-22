from config.config_wgan import *
from torch import nn


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, num_classes):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(
            # Input: batch x z_dim x 1 x 1 (see definition of noise in below code)
            self._block(z_dim + embed_size + feature_size, gen_dim * 16, 4, 1, 0),  # batch x 1024 x 4 x 4
            self._block(gen_dim * 16, gen_dim * 8, 4, 2, 1),  # batch x 512 x 8 x 8
            self._block(gen_dim * 8, gen_dim * 4, 4, 2, 1),  # batch x 256 x 16 x 16
            self._block(gen_dim * 4, gen_dim * 2, 4, 2, 1),  # batch x 128 x 32 x 32
            nn.ConvTranspose2d(
                gen_dim * 2, num_channel_img, kernel_size=4, stride=2, padding=1,
                # did not use block because the last layer won't use batch norm or relu
            ),  # batch x 3 x 64 x 64
            nn.Tanh(),
            # squeeze output to [-1, 1]; easier to converge.  also will match to our normalize(0.5....) images
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)  # in_place = True
        )

    def forward(self, x, labels, semantic_latent):
        # semantic latent: batch, feature_size
        # Input: latent vector z: batch x z_dim x 1 x 1
        # in order to concat labels with the latent vector, we have to create two more dimensions of 1 by unsqueezing
        semantic_latent = semantic_latent.unsqueeze(2).unsqueeze(3)  # batch, feature_size, 1, 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)  # batch, embed_size, 1, 1
        x = torch.cat([x, embedding, semantic_latent], dim=1)
        return self.net(x)


class Discriminator2(nn.Module):
    def __init__(self, ngpu, num_classes):
        super(Discriminator2, self).__init__()
        self.ngpu = ngpu
        self.latent_joining = nn.Sequential(
            nn.Linear(feature_size, image_size * image_size)
        )
        self.net = nn.Sequential(
            # no batch norm in the first layer
            # Input: batch x num_channel x 64 x 64
            # <-----changed num_channel + 1 since we add the labels
            nn.Conv2d(
                num_channel_img + 2, dis_dim, kernel_size=4, stride=2, padding=1,
            ),  # batch x 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(dis_dim, dis_dim * 2, 4, 2, 1),  # batch x 128 x 16 x 16
            self._block(dis_dim * 2, dis_dim * 4, 4, 2, 1),  # batch x 256 x 8 x 8
            self._block(dis_dim * 4, dis_dim * 8, 4, 2, 1),  # batch x 512 x 4  x 4
            nn.Conv2d(dis_dim * 8, 1, kernel_size=4, stride=2, padding=0),  # batch x 1 x 1 x 1 for classification
            #             nn.Sigmoid(), #<------removed!
        )
        self.embed = nn.Embedding(num_classes, image_size * image_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.InstanceNorm2d(out_channels, affine=True),  # <----changed here
            nn.LeakyReLU(0.2, True)  # slope = 0.2, in_place = True
        )

    def forward(self, x, feature, labels):
        # Label shape: batch,
        # Label after embed shape: batch, image_size * image_size
        # reshape the labels further to be of shape (batch, 1, H, W) so we can concat
        # embedding shape:  batch, 1, image_size, image_size
        feature_plate = self.latent_joining(feature).view(feature.shape[0], 1, image_size, image_size)
        embedding = self.embed(labels).view(labels.shape[0], 1, image_size, image_size)

        # feature_em = self.embed_feature(feature).view(feature.shape[0], 1, image_size, image_size)
        x = torch.cat([x, feature_plate, embedding], dim=1)  # batch x (C + 1) x W x H
        return self.net(x)


class AlexNetExtractor(nn.Module):
    """
    This class expected image as input with size (64x64x3)
    """

    def __init__(self, output_class_num, in_channel=3, feature_size=200, pretrain=False):
        super(AlexNetExtractor, self).__init__()
        self.feature_size = feature_size
        self.num_classes = output_class_num
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

        self._add_classifier(self.num_classes, self.feature_size)

    def _add_classifier(self, num_classes, feature_size):
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
        self.fc08 = nn.Sequential(
            nn.Linear(feature_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc06(x)
        semantic_features = self.fc07(x)
        p_label = self.fc08(semantic_features)
        return semantic_features, p_label
