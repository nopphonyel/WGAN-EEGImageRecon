from module import *
import math


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
            raise "<X> \'qkv_dim\' must be divisible by the \'head_num\'"
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


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, in_channel, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, in_channel)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channel, 2) * -(math.log(10000.0) / in_channel))
        pe[:, 0::2] = torch.sin(position * div_term)[:, 0:pe[:, 0::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * div_term)[:, 0:pe[:, 1::2].shape[1]]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
