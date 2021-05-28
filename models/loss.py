import torch
import torch.nn as nn

def getloss(type):
    if type == 'l1':
        return nn.L1Loss()
    elif type == 'l2':
        return nn.MSELoss()