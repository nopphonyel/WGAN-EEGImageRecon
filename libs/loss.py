import torch


def j1_loss(l1, l2, f1, f2):
    """
    This loss implementation is following Dan Li et al works.
    According to the Dan Li et al. implementation:
        J1 (f1, f2) = (non-img latent, img_paired latent)
        J4 (f1, f2) = (non-img latent, img_unpaired latent)
    :param l1: A one hot encoded label
    :param l2: Another one hot encoded label
    :param f1: A latent (From paper, they use non-image) but I think both can be swap
    :param f2: Another latent
    :return:
    """
    s = torch.matmul(l2, l1.transpose(1, 0))
    delta = 0.5 * torch.matmul(torch.tanh(f1), torch.tanh(f2.transpose(1, 0)))
    losses = torch.mul(s, delta) - torch.log(torch.exp(delta))
    loss = torch.mean(losses)
    return loss
