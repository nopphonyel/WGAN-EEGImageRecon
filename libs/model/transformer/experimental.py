from preimport_module import *


class SplitQKVAttention(nn.Module):
    """
    Expected Input Shape: (batch, seq_len, channels)
    """

    def __init__(self, qkv_dim, dropout_p=0.0):
        """
        A multi-head attention module
        :param qkv_dim: Size of input channel
        :param head_num: Number of head being used (must be divisible by head_num)
        :param drophead_p: currently unavailable
        """
        super(SplitQKVAttention, self).__init__()

        self.softmax = nn.LogSoftmax(dim=1)

        self.lin_Q = nn.Linear(qkv_dim, qkv_dim, bias=False)
        self.lin_K = nn.Linear(qkv_dim, qkv_dim, bias=False)
        self.lin_V = nn.Linear(qkv_dim, qkv_dim, bias=False)
        self.lin_WO = nn.Linear(qkv_dim, qkv_dim, bias=False)

        self.dropout_p = dropout_p

        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, q, k, v):
        # lstm_output.shape(out)  =  ([256, 491, 64])
        # final_state.shape(hn)   =  ([256, 64])

        q = self.lin_Q(torch.clone(q))
        k = self.lin_K(torch.clone(k))
        v = self.lin_V(torch.clone(v))

        qk = torch.matmul(q, k.transpose(1, 2))
        sftmx_qk = self.softmax(qk)
        if self.dropout_p > 0:
            sftmx_qk = self.dropout(sftmx_qk)
        attn = torch.matmul(sftmx_qk, v)

        return self.lin_WO(attn)
