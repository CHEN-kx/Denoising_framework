import torch.nn as nn
from models.utils import *
from models.common import *

class SRVC(nn.Module):
    def __init__(self, cfg):
        super(SRVC, self).__init__()

        self.k = cfg.k
        self.adaconv_nc = cfg.adaconv_nc
        self.rgb_ic, self.af_ic = cfg.input_nc_x, cfg.input_nc_ft
        self.inchannles = self.rgb_ic + self.af_ic
        self.s2b = Space2Batch(cfg.spacesize)

        # define head
        norm_layer = get_norm_layer(cfg.norm_type)
        self.head = nn.Sequential(*[valid_conv(self.inchannles, cfg.head_nc, kernel_size=3),
                                norm_layer(8),
                                nn.ReLU(inplace=True)])
        self.head.apply(init_weights)

        # define adaConv
        self.kernel = BasicBlock(valid_conv, cfg.head_nc, 9*self.inchannles*cfg.adaconv_nc, 3, bias=True, bn=False, act=None)
        self.bias = BasicBlock(valid_conv, cfg.head_nc, cfg.adaconv_nc, 3, bias=True, bn=False, act=None)

        self.b2s = Batch2Space(cfg.spacesize)

        # define body
        m_body = [IdentityResBlock(default_conv, cfg.adaconv_nc, cfg.out_nc, 5, bias=True, bn=False)]
        if self.k==0:
            m_body.append(IdentityResBlock(default_conv, cfg.out_nc, 3, 3, bias=True, bn=False, act=None))
        else:
            m_body.append(IdentityResBlock(default_conv, cfg.out_nc, self.k**2, 3, bias=True, bn=False, act=None))
        self.pred_net = nn.Sequential(*m_body)
        self.pred_net.apply(init_weights)

    def forward(self, input):
        X, ft = input
        # full -> 5*5 (including padding)
        x, B,C,H,W,pad_h,pad_w = self.s2b(torch.cat((X, ft), dim=1))
        # 5*5 -> 3*3
        feature = self.head(x)

        # 3*3 -> 1*1
        kernel = self.kernel(feature).view(-1, self.adaconv_nc, self.inchannles, 3, 3) # (B, out_C, in_C, iH, iW)
        bias = self.bias(feature).view(-1, self.adaconv_nc) # (B, out_C)
        #pdb.set_trace()
        # 5*5 -> 5*5
        x = adaConv(x, kernel, bias)      

        # 5*5 -> full
        x = self.b2s(x,B,C,H+pad_h,W+pad_w)
        x = x[:,:,:H,:W]
        
        if self.k==0:
            return self.pred_net(x)
        else:
            return apply_kernel(X, self.pred_net(x), self.k, normalize=True)  