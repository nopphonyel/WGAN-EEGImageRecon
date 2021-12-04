from libs.model.transformer.component import *
from libs.model.transformer.experimental import *


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



