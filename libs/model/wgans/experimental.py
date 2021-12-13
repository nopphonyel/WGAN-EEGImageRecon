from config.config_wgan import *
from module import *


# Generator Code
class GeneratorNoLabel(nn.Module):
    def __init__(self, ngpu, num_classes, latent_size,  img_channel=3):
        super(GeneratorNoLabel, self).__init__()
        self.ngpu = ngpu
        self.img_channel = img_channel
        self.net = nn.Sequential(
            # Input: batch x (z_dim + embed_size + feature_size) x 1 x 1 (see definition of noise in below code)
            self._block(z_dim + latent_size, gen_dim * 16, 4, 1, 0),  # batch x 1024 x 4 x 4
            self._block(gen_dim * 16, gen_dim * 8, 4, 2, 1),  # batch x 512 x 8 x 8
            self._block(gen_dim * 8, gen_dim * 4, 4, 2, 1),  # batch x 256 x 16 x 16
            self._block(gen_dim * 4, gen_dim * 2, 4, 2, 1),  # batch x 128 x 32 x 32
            nn.ConvTranspose2d(
                gen_dim * 2, self.img_channel, kernel_size=4, stride=2, padding=1,
                # did not use block because the last layer won't use batch norm or relu
            ),  # batch x 3 x 64 x 64
            nn.Tanh(),
            # squeeze output to [-1, 1]; easier to converge.  also will match to our normalize(0.5....) images
        )
        # self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)  # in_place = True
        )

    def forward(self, zz, semantic_latent):
        # semantic latent: batch, feature_size
        # Input: latent vector z: batch x z_dim x 1 x 1
        # in order to concat labels with the latent vector, we have to create two more dimensions of 1 by unsqueezing
        semantic_latent = semantic_latent.unsqueeze(2).unsqueeze(3)  # batch, feature_size, 1, 1
        x = torch.cat([zz, semantic_latent], dim=1)
        return self.net(x)


class DiscriminatorNoLabel(nn.Module):
    def __init__(self, ngpu, num_classes, latent_size, img_channel=3):
        super(DiscriminatorNoLabel, self).__init__()
        self.ngpu = ngpu
        self.img_channel = img_channel
        self.latent_joining = nn.Sequential(
            nn.Linear(latent_size, image_size * image_size)
        )
        self.net = nn.Sequential(
            # no batch norm in the first layer
            # Input: batch x num_channel x 64 x 64
            # <-----changed num_channel + 1 since we add the labels
            nn.Conv2d(
                self.img_channel + 1, dis_dim, kernel_size=4, stride=2, padding=1,
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

    def forward(self, x, feature):
        # Label shape: batch,
        # Label after embed shape: batch, image_size * image_size
        # reshape the labels further to be of shape (batch, 1, H, W) so we can concat
        # embedding shape:  batch, 1, image_size, image_size
        feature_plate = self.latent_joining(feature).view(feature.shape[0], 1, image_size, image_size)

        # feature_em = self.embed_feature(feature).view(feature.shape[0], 1, image_size, image_size)
        x = torch.cat([x, feature_plate], dim=1)  # batch x (C + 1) x W x H
        return self.net(x)
