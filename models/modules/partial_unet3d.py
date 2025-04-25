import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .partialconv3d import *
from .module import *


def norm_data(data):
    mean = torch.mean(data,dim=(-3,-2,-1), keepdim=True)
    std = torch.std(data,dim=(-3,-2,-1), keepdim=True)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def norm_data_out(normalized_data, mean, std):
    data = normalized_data * std + mean
    return data

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def norm_layer(name, out_channels):
    if name == 'bn':
        return nn.BatchNorm3d(out_channels)
    elif name == 'in':
        return nn.InstanceNorm3d(out_channels)

# --------------------------
# PConv-BatchNorm-Activation
# --------------------------
class PConvBNActiv(nn.Module):
    def __init__(self, in_channels, out_channels, norm, sample='none-3', activ='relu', bias=False):
        super(PConvBNActiv, self).__init__()
        if sample == 'down-7':
            self.conv = PartialConv3d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias, return_mask=True, multi_channel = True)
        elif sample == 'down-5':
            self.conv = PartialConv3d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=bias, return_mask=True, multi_channel = True)
        elif sample == 'down-3':
            self.conv = PartialConv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias, return_mask=True, multi_channel = True)
        else:
            self.conv = PartialConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, return_mask=True, multi_channel = True)
        if norm:
            self.norm = norm_layer(norm, out_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, images, masks):
        images, masks = self.conv(images, masks)
        if hasattr(self, 'norm'):
            images = self.norm(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        return images, masks

# ------------
# Double U-Net
# ------------
class PUNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, actv, mode, up_sampling_node='nearest', init_weight=False):
        super(PUNet, self).__init__()
        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node
        self.ec_images_1 = PConvBNActiv(in_channels, mid_channels, norm=None, sample='down-7', activ='leaky')
        self.ec_images_2 = PConvBNActiv(mid_channels*1, mid_channels*2, norm='in', sample='down-5', activ='leaky')
        self.ec_images_3 = PConvBNActiv(mid_channels*2, mid_channels*4, norm='in', sample='down-3', activ='leaky')
        self.ec_images_4 = PConvBNActiv(mid_channels*4, mid_channels*8, norm='in', sample='down-3', activ='leaky')

        self.dc_images_4 = PConvBNActiv(mid_channels*8 + mid_channels*4, mid_channels*4, norm='in', activ='leaky')
        self.dc_images_3 = PConvBNActiv(mid_channels*4 + mid_channels*2, mid_channels*2, norm='in', activ='leaky')
        self.dc_images_2 = PConvBNActiv(mid_channels*2 + mid_channels*1, mid_channels*1, norm='in', activ='leaky')
        self.dc_images_1 = PConvBNActiv(mid_channels*1 + out_channels, out_channels, norm=None, sample='none-3', activ=None, bias=True)
        self.non_nega = nn.ReLU(inplace=True)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, tbe=None, ten=None):
        input_images = x[:,:1,tbe:ten,...]
        input_masks  = x[:,1:,tbe:ten,...]
        
        ec_images = {}
        ec_images['ec_images_0'], ec_images['ec_images_masks_0'] = input_images, input_masks
        ec_images['ec_images_1'], ec_images['ec_images_masks_1'] = self.ec_images_1(input_images, input_masks)
        ec_images['ec_images_2'], ec_images['ec_images_masks_2'] = self.ec_images_2(ec_images['ec_images_1'], ec_images['ec_images_masks_1'])
        ec_images['ec_images_3'], ec_images['ec_images_masks_3'] = self.ec_images_3(ec_images['ec_images_2'], ec_images['ec_images_masks_2'])
        ec_images['ec_images_4'], ec_images['ec_images_masks_4'] = self.ec_images_4(ec_images['ec_images_3'], ec_images['ec_images_masks_3'])

        # --------------
        # images decoder
        # --------------
        dc_images, dc_images_masks = ec_images['ec_images_4'], ec_images['ec_images_masks_4']
        for _ in range(4, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            ec_images_masks = 'ec_images_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)
            dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images_masks = F.interpolate(dc_images_masks, scale_factor=2, mode=self.up_sampling_node)
            dc_images = torch.cat((dc_images, ec_images[ec_images_skip]), dim=1)
            dc_images_masks = torch.cat((dc_images_masks, ec_images[ec_images_masks]), dim=1)
            dc_images, dc_images_masks = getattr(self, dc_conv)(dc_images, dc_images_masks)
            
        outputs = self.non_nega(dc_images+ec_images['ec_images_0'])
        
        denoise_transients = torch.zeros_like(x[:,:1,...]).to(x.device)
        denoise_transients[:,:,tbe:ten,...] = outputs
        
        return denoise_transients
