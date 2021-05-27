import torch.nn as nn
from models.utils import *
from models.common import *
from models.backbone import ResNet

class KPCN(nn.modules):
    def __init__(self, cfg):
        super(KPCN, self).__init__()
        norm_layer = get_norm_layer(cfg.norm_type)
        self.k = cfg.k
        self.basic_net = ResNet(cfg.input_nc, cfg.hidden_nc, cfg.n_blocks, cfg.norm_type)

        self.pred_net = nn.Sequential(*[default_conv(cfg.hidden_nc, cfg.hidden_nc, kernel_size=1),
                                        norm_layer(cfg.hidden_nc), nn.ReLU(),
                                        default_conv(cfg.hidden_nc, self.k ** 2, kernel_size=1)])
        self.pred_net.apply(init_weights)

    def forward(self, x):
        kernel = self.pred_net(self.basic_net(x))
        y = apply_kernel(x[0], kernel, self.k, normalize=True)

        return y