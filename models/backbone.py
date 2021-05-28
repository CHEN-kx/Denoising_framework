import torch
import torch.nn as nn

from models.utils import init_weights
from models.common import *

class ResNet(nn.Module):
    def __init__(self, input_nc, hidden_nc, n_blocks, norm_type='none'):
        super(ResNet, self).__init__()
        self.x_nc, self.ft_nc = input_nc[0], input_nc[1]
        self.input_nc = self.x_nc + self.ft_nc
        norm_layer = get_norm_layer(norm_type)

        self.head = nn.Sequential(*[default_conv(self.input_nc, hidden_nc, 3),
                                    norm_layer(hidden_nc),relu(inplace=True)])
        self.head.apply(init_weights)

        self.body_list=[ResBlock(default_conv, hidden_nc, 3, norm_type) for _ in range(n_blocks)]
        self.body = nn.Sequential(*self.body_list)

    def forward(self, x):
        x_ft = torch.cat((x[0], x[1]), dim=1)
        
        return self.body(self.head(x_ft))


class UNet(nn.Module):
    ''' oidn '''
    def __init__(self, input_nc, output_nc, norm_type='none'):
        # Number of channels per layer
        ic = input_nc
        ec1 = 32
        ec2 = 48
        ec3 = 64
        ec4 = 80
        ec5 = 96
        dc4 = 112
        dc3 = 96
        dc2 = 64
        dc1a = 64
        dc1b = 32
        oc = output_nc        

        # Convolutions
        self.enc_conv0 = default_conv(ic, ec1, 3)
        self.enc_conv1 = default_conv(ec1, ec1, 3)
        self.enc_conv2 = default_conv(ec1, ec2, 3)
        self.enc_conv3 = default_conv(ec2, ec3, 3)
        self.enc_conv4 = default_conv(ec3, ec4, 3)
        self.enc_conv5a = default_conv(ec4, ec5, 3)
        self.enc_conv5b = default_conv(ec5, ec5, 3)
        self.dec_conv4a = default_conv(ec5 + ec3, dc4, 3)
        self.dec_conv4b = default_conv(dc4, dc4, 3)
        self.dec_conv3a = default_conv(dc4 + ec2, dc3, 3)
        self.dec_conv3b = default_conv(dc3, dc3, 3)
        self.dec_conv2a = default_conv(dc3 + ec1, dc2, 3)
        self.dec_conv2b = default_conv(dc2, dc2, 3)
        self.dec_conv1a = default_conv(dc2 + ic, dc1a, 3)
        self.dec_conv1b = default_conv(dc1a, dc1b, 3)
        self.dec_conv0 = default_conv(dc1b, oc, 3)

        # Images must be padded to multiples of the alignment
        self.alignment = 16

    def forward(self, input):
        # Encoder
        # -------------------------------------------
        x = relu(self.enc_conv0(input))  # enc_conv0
        x = relu(self.enc_conv1(x))  # enc_conv1
        x = pool1 = pool(x)  # pool1
        x = relu(self.enc_conv2(x))  # enc_conv2
        x = pool2 = pool(x)  # pool2
        x = relu(self.enc_conv3(x))  # enc_conv3
        x = pool3 = pool(x)  # pool3
        x = relu(self.enc_conv4(x))  # enc_conv4
        x = pool(x)  # pool4

        # Bottleneck
        x = relu(self.enc_conv5a(x))  # enc_conv5a
        x = relu(self.enc_conv5b(x))  # enc_conv5b

        # Decoder
        # -------------------------------------------
        x = upsample(x, pool3)  # upsample4
        x = concat(x, pool3)  # concat4
        x = relu(self.dec_conv4a(x))  # dec_conv4a
        x = relu(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x, pool2)  # upsample3
        x = concat(x, pool2)  # concat3
        x = relu(self.dec_conv3a(x))  # dec_conv3a
        x = relu(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x, pool1)  # upsample2
        x = concat(x, pool1)  # concat2
        x = relu(self.dec_conv2a(x))  # dec_conv2a
        x = relu(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x, input)  # upsample1
        x = concat(x, input)  # concat1
        x = relu(self.dec_conv1a(x))  # dec_conv1a
        x = relu(self.dec_conv1b(x))  # dec_conv1b

        x = self.dec_conv0(x)  # dec_conv0

        return x