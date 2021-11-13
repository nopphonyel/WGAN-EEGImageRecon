import torch

Q = torch.randn([1, 440, 128])
K = torch.randn([1, 440, 128])
V = torch.randn([1, 440, 128])

q = Q.reshape([1, 440, 8, 16])
k = K.reshape([1, 440, 8, 16])
v = V.reshape([1, 440, 8, 16])

qk = torch.matmul(q, k.transpose(2, 3))
attn = torch.matmul(qk, v)
print(attn.shape)
