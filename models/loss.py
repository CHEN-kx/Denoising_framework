import torch
import torch.nn as nn

def getloss(cfg):
    if cfg.loss == 'l1':
        return nn.L1Loss()
    elif cfg.loss == 'l2':
        return nn.MSELoss()