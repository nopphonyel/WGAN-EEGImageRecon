import torch

from preimport_module import *
import torch.linalg as la


# There is a replacement -> nn.ConsineSimilarity
def sim(a: torch.Tensor, b: torch.Tensor):
    na = la.matrix_norm(a)
    nb = la.matrix_norm(b)
    return
