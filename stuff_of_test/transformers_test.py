import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

device = "cuda"
hidden_dim = 64


class MultiHeadAttention(nn.Module):
    """
    Expected Input Shape: (batch, seq_len, channels)
    """

    def __init__(self, input_dim, head_num, drophead_p=0.0, dropout_p=0.0):
        """
        A multi-head attention module
        :param input_dim: Size of input channel
        :param head_num: Number of head being used (must be divisible by head_num)
        :param drophead_p: currently unavailable
        """
        super(MultiHeadAttention, self).__init__()

        self.softmax = nn.LogSoftmax(dim=1)

        self.head_num = head_num

        if input_dim % head_num != 0:
            raise "<X> \'input_dim\' must be divisible by the \'head_num\'"
        self.head_size = input_dim // self.head_num
        self.head_size_t = nn.Parameter(torch.Tensor([self.head_size]), requires_grad=False)

        self.lin_Q = nn.Linear(input_dim, input_dim, bias=False)
        self.lin_K = nn.Linear(input_dim, input_dim, bias=False)
        self.lin_V = nn.Linear(input_dim, input_dim, bias=False)
        self.lin_WO = nn.Linear(input_dim, input_dim, bias=False)

        self.drophead_p = drophead_p
        self.dropout_p = dropout_p

        if drophead_p > 0:
            self.drophead = nn.Dropout(p=drophead_p)
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # lstm_output.shape(out)  =  ([256, 491, 64])
        # final_state.shape(hn)   =  ([256, 64])
        bs = x.shape[0]
        ln = x.shape[1]

        q = self.lin_Q(torch.clone(x))
        k = self.lin_K(torch.clone(x))
        v = self.lin_V(torch.clone(x))

        # mha = Multi Head Attention
        q = q.reshape((bs, ln, self.head_num, self.head_size))
        k = k.reshape((bs, ln, self.head_num, self.head_size))
        v = v.reshape((bs, ln, self.head_num, self.head_size))

        qk = torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(self.head_size_t)
        sftmx_qk = self.softmax(qk)
        if self.dropout_p > 0:
            sftmx_qk = self.dropout(sftmx_qk)
        attn = torch.matmul(sftmx_qk, v)

        # Drop head performs here
        if self.drophead_p > 0:
            dev = next(self.parameters()).device
            msk = torch.ones([bs, self.head_num]).to(dev)
            msk = torch.sign(self.drophead(msk))
            msk = msk.reshape([bs, 1, self.head_num, 1])
            attn = msk * attn

        attn = attn.reshape((bs, ln, self.head_num * self.head_size))
        return self.lin_WO(attn)


class EEGTransformerEncoder(nn.Module):
    """
        Expected Input Shape: (batch, seq_len, channels)
    """

    def __init__(self, input_dim, head_num, drophead_p=0.0, dropout_p=0.0, len_reduction=None):
        """

        :param input_dim: Size or number of input features
        :param head_num: Number of head in MultiHeadAttention module
        :param drophead_p : Probability that each head being drop (Bernoulli distribution)
        :param len_reduction: Eliminate the seq_len dimension by 'sum' or 'mean' operation
        """
        super(EEGTransformerEncoder, self).__init__()

        self.len_reduction = len_reduction
        self.multi_head = MultiHeadAttention(input_dim, head_num, drophead_p=drophead_p, dropout_p=dropout_p)
        self.ly_norm1 = nn.LayerNorm(input_dim)
        self.ff = nn.Linear(input_dim, input_dim)
        self.ly_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x_orig):
        x = self.multi_head(x_orig)
        x_orig = self.ly_norm1(x_orig + x)
        x = self.ff(x_orig)
        x = self.ly_norm2(x_orig + x)

        if self.len_reduction == "mean":
            return torch.mean(x, dim=1)
        elif self.len_reduction == "sum":
            return torch.mean(x, dim=1)
        elif self.len_reduction is None:
            return x
        else:
            raise "<X> Unknown len_reduction mode... Consider using \'mean\' or \'sum\'"


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


class CustomModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomModel, self).__init__()
        self.pe = PositionalEncoding(input_dim, dropout=0)
        self.enc1 = EEGTransformerEncoder(input_dim, head_num=8)
        self.enc2 = EEGTransformerEncoder(input_dim, head_num=16, len_reduction='sum')
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.pe(x)
        x = self.enc1(x)
        x = self.enc2(x)
        return self.fc(x)


BS = 1
CHANNEL = 128
LEN = 440

a_tensor = torch.randn([BS, LEN, CHANNEL]).to(device)
model = CustomModel(CHANNEL, 40).to(device)
out = model(a_tensor)
print(out.shape)
