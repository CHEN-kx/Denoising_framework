import torch
import torch.nn as nn
from torch.functional import norm

from models.utils import init_weights
from models.common import *

class ResNet(nn.Module):
    def __init__(self, input_nc, hidden_nc, n_blocks, norm_type='none'):
        super(ResNet, self).__init__()
        self.x_nc, self.ft_nc = input_nc[0], input_nc[1]
        self.input_nc = self.x_nc + self.ft_nc
        norm_layer = get_norm_layer(norm_type)

        self.head = nn.Sequential(*[default_conv(self.input_nc, hidden_nc, 3),
                                    norm_layer(hidden_nc),nn.ReLU(inplace=True)])
        self.head.apply(init_weights)

        self.body_list=[ResBlock(default_conv, hidden_nc, 3, norm_type) for _ in range(n_blocks)]
        self.body = nn.Sequential(*self.body_list)

    def forward(self, x):
        x_ft = torch.cat((x[0], x[1]), dim=1)
        
        return self.body(self.head(x_ft))
