from module import *


class StupidFC01(nn.Module):
    """
    Expected input shape = [bs, 948]
    """

    def __init__(self, latent_size, num_classes, dropout_p=0.2):
        super(StupidFC01, self).__init__()

        self.block01 = nn.Sequential(
            nn.BatchNorm1d(948),
            nn.Linear(948, 470, bias=False),
            nn.Dropout(p=dropout_p),
            nn.LeakyReLU()
        )

        self.block02 = nn.Sequential(
            nn.BatchNorm1d(470),
            nn.Linear(470, latent_size),
            nn.LeakyReLU()
        )

        self.block03 = nn.Sequential(
            nn.BatchNorm1d(latent_size),
            nn.Linear(latent_size, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.block01(x)
        latent = self.block02(x)
        out = self.block03(latent)
        return latent, out
