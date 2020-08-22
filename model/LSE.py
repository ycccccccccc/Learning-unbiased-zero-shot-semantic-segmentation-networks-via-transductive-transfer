import torch


def lse(output):
    r = 5
    hw = output.shape[2] * output.shape[3]
    x = output * r
    x = torch.exp(x)
    x = torch.sum(x,[2,3])
    x = x / hw + 0.000001
    x = torch.log(x)
    x = x / r
    return x
