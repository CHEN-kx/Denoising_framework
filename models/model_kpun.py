from models.backbone import UNet
from models.common import *

class KPUN(nn.Module):
    """ kpn U-Net model """
    def __init__(self, cfg):
        super(KPUN, self).__init__()
        self.k = cfg.k
        self.unet = UNet(in_channels=sum(cfg.input_nc), out_channels=cfg.k * cfg.k)

    def forward(self, input):
        x, ft = input[0], input[1]
        x_ft = concat(x, ft)
        kernel = self.unet(x_ft)

        return apply_kernel(x, kernel, self.k, normalize=True)  