import math
import torch
from torch.autograd import Variable
from torch import nn

import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, in_channel, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, in_channel)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channel, 2) *
                             -(math.log(10000.0) / in_channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


BS = 1
CHANNEL = 128
LEN = 440

device = "cuda"

pe = PositionalEncoding(in_channel=CHANNEL, dropout=0, max_len=LEN).to(device)
eeg = torch.randn([BS, LEN, CHANNEL]).to(device)

pe_eeg = pe(eeg)
print(pe_eeg.shape)
