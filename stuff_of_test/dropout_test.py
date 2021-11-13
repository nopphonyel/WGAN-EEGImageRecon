import torch
from torch import nn

head_num = 3
head_size = 4
BS = 2
attn = torch.randn([BS, 5, head_num, head_size])

drop_out = nn.Dropout(0.2)
msk = torch.ones([BS, head_num])
msk = drop_out(msk)
print(torch.sign(msk))
msk = msk.reshape([BS, 1, head_num, 1])

a_tensor_drp_out = msk * attn
print(a_tensor_drp_out)
