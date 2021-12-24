from preimport_module import *
from libs.model.transformer import *


class LSTMandAttention(nn.Module):
    def __init__(self, data_len, latent_size, num_classes):
        super(LSTMandAttention, self).__init__()
        self.dev = 'cpu'
        self.lstm = nn.LSTM(
            input_size=1,
            batch_first=True,
            dropout=0.1,
            hidden_size=3,
            num_layers=1,
            bias=True
        )
        self.attn = SplitQKVAttention(qkv_dim=1)

        self.fc_latent = nn.Linear(data_len, latent_size, bias=False)
        self.fc_final = nn.Linear(latent_size, num_classes, bias=False)

    def to(self, dev):
        super().to(dev)
        self.dev = dev
        return self

    def forward(self, x):
        x = LSTMandAttention.__input_reshape(x)
        h0, c0 = torch.randn(1, x.shape[0], 3, device=self.dev), \
                 torch.randn(1, x.shape[0], 3, device=self.dev)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        q = x[:, :, 0].unsqueeze(2)
        k = x[:, :, 1].unsqueeze(2)
        v = x[:, :, 2].unsqueeze(2)
        x = self.attn(q, k, v).squeeze(2)
        latent = self.fc_latent(x)
        out = self.fc_final(latent)
        return latent, out

    @staticmethod
    def __input_reshape(x):
        return x.unsqueeze(2)


class BiLSTMMultihead(nn.Module):
    """
    Expected with 1 channel FMRI data, shape = (BS, FMRI_DATA)
    """

    def __init__(
            self,
            num_classes: int,
            latent_size: int,
            data_len: int,
            lstm_dropout: float = 0,
            lstm_hidden_size: int = 4,
            lstm_num_layers: int = 1,
            lstm_bias: bool = False,
            multihead_dropout_p: float = 0.1
    ):
        super(BiLSTMMultihead, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=1,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bias=lstm_bias
        )
        self.multiHead = MultiHeadAttention(
            input_dim=self.lstm_hidden_size * 2,
            head_num=self.lstm_hidden_size,
            dropout_p=multihead_dropout_p
        )
        self.calc_dim = data_len * self.lstm_hidden_size * 2
        self.latent_fc = nn.Sequential(
            nn.Linear(in_features=self.calc_dim, out_features=latent_size),
            nn.LeakyReLU()
        )

        self.final_fc = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=num_classes)
        )

        self.dev = 'cpu'

    def to(self, dev):
        super().to(dev)
        self.dev = dev
        return self

    def forward(self, x):
        x = BiLSTMMultihead.__input_reshape(x)
        h0, c0 = torch.randn(2, x.shape[0], self.lstm_hidden_size, device=self.dev), \
                 torch.randn(2, x.shape[0], self.lstm_hidden_size, device=self.dev)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.multiHead(x)
        x = x.flatten(start_dim=1)
        latent = self.latent_fc(x)
        out = self.final_fc(latent)

        return latent, out

    @staticmethod
    def __input_reshape(x):
        return x.unsqueeze(2)


class BiLSTMMultihead2xEncoder(nn.Module):

    def __init__(
            self,
            num_classes: int,
            latent_size: int,
            data_len: int,
            lstm_dropout: float = 0,
            lstm_hidden_size: int = 4,
            lstm_num_layers: int = 1,
            lstm_bias: bool = False,
            multihead_dropout_p: float = 0.1
    ):
        super(BiLSTMMultihead2xEncoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=1,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bias=lstm_bias
        )
        self.multiHead = nn.Sequential(
            MultiHeadAttention(
                input_dim=self.lstm_hidden_size * 2,
                head_num=self.lstm_hidden_size,
                dropout_p=multihead_dropout_p
            ),
            MultiHeadAttention(
                input_dim=self.lstm_hidden_size * 2,
                head_num=self.lstm_hidden_size,
                dropout_p=multihead_dropout_p
            ),
        )

        self.calc_dim = data_len * self.lstm_hidden_size * 2
        self.latent_fc = nn.Sequential(
            nn.Linear(in_features=self.calc_dim, out_features=latent_size),
            nn.LeakyReLU()
        )

        self.final_fc = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=num_classes)
        )

        self.dev = 'cpu'

    def to(self, dev):
        super().to(dev)
        self.dev = dev
        return self

    def forward(self, x):
        x = BiLSTMMultihead2xEncoder.__input_reshape(x)
        h0, c0 = torch.randn(2, x.shape[0], self.lstm_hidden_size, device=self.dev), \
                 torch.randn(2, x.shape[0], self.lstm_hidden_size, device=self.dev)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.multiHead(x)
        x = x.flatten(start_dim=1)
        latent = self.latent_fc(x)
        out = self.final_fc(latent)

        return latent, out

    @staticmethod
    def __input_reshape(x):
        return x.unsqueeze(2)


class FCAndMultihead(nn.Module):
    def __init__(self, input_size, head_num, latent_size, num_classes):
        super(FCAndMultihead, self).__init__()
        self.mhead_1 = MultiHeadAttention(input_dim=input_size, head_num=head_num, dropout_p=0.1)
        self.fc1 = nn.Sequential(
            nn.Linear(948, 480),
            nn.LeakyReLU()
        )
        self.mhead_2 = MultiHeadAttention(input_dim=480, head_num=head_num, dropout_p=0.1)
        self.fc2 = nn.Sequential(
            nn.Linear(480, latent_size),
            nn.LeakyReLU()
        )
        # self.mhead_3 = MultiHeadAttention(qkv_dim=latent_size, head_num=head_num, dropout_p=0.1)
        # self.fc_latent = nn.Sequential(
        #     nn.Linear(latent_size, latent_size),
        #     nn.LeakyReLU()
        # )
        self.fc3 = nn.Sequential(
            nn.Linear(latent_size, num_classes),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mhead_1(x)
        x = self.fc1(x)
        x = self.mhead_2(x)
        latent = self.fc2(x)
        out = self.fc3(latent)
        return latent.squeeze(1), out.squeeze(1)


class DumbAssFC(nn.Module):
    def __init__(self, latent_size, num_classes):
        super(DumbAssFC, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(948, latent_size),
            nn.LeakyReLU()
        )

        self.block_last = nn.Sequential(
            nn.Linear(latent_size, num_classes),
        )

    def forward(self, x):
        latent = self.block(x)
        out = self.block_last(latent)
        return latent, out


class Conv1DStuff(nn.Module):
    def __init__(self, data_len, latent_size, num_classes):
        super(Conv1DStuff, self).__init__()
        self.conv1d01 = nn.Sequential(
            nn.BatchNorm1d(num_features=1),
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=data_len // 4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=data_len // 8, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=4),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=data_len // 16),
            nn.LeakyReLU()
        )

        self.fc_latent = nn.Sequential(
            nn.Linear(496, latent_size, bias=False)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(latent_size, num_classes)
        )

    def forward(self, x):
        x = Conv1DStuff.__input_reshape(x)
        x = self.conv1d01(x)
        latent = self.fc_latent(x.flatten(start_dim=1))
        out = self.fc_out(latent)

        return latent, out

    @staticmethod
    def __input_reshape(x):
        return x.unsqueeze(1)


if __name__ == '__main__':
    BS = 2
    FF = 948
    t = torch.randn(BS, FF)
    m = LSTMandAttention(num_classes=6, latent_size=200, data_len=FF)
    ol, oo = m(t)
    print(ol.shape, oo.shape)
