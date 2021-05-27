import torch
import torch.nn as nn
import torch.nn.init as init

def crop_input(data, krl, k):
    """crop data as krl"""
    # data: (b,c,h,w), krl: (b,k**2,h_,w_)
    r = k // 2
    if data.shape[-2:] != krl.shape[-2:]:
        with torch.no_grad():
            dx = data.shape[-2] - krl.shape[-2] - 2 * r
            dy = data.shape[-1] - krl.shape[-1] - 2 * r
            data = data[:, :, dx // 2: - dx // 2, dy // 2: - dy // 2]
    return data

def init_weights(m):
    """init weight"""
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)  # weight init
        init.constant_(m.bias.data, 0.0)  # bias init


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """xavier init"""
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """normal init"""
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)