import torch
import torch.nn as nn
from .physics.lct import LCT
from .modules.unet3d_attention import unet3d_atte
from .modules.partial_unet3d import PUNet
import copy


class VSmodel(nn.Module):
    def __init__(self, cfg, type, device, physics=True):
        super(VSmodel, self).__init__()
        
        if physics:
            self.Physics = Physics_model(cfg, device)
        self.Neural = Net(cfg, type, device)
    
    def neural(self, x, tbe, ten):
        return self.Neural(x, tbe, ten)
    
    def forward_operator(self, x, tbe, ten):
        out = self.Physics.forward_operator(x, tbe, ten)
        return out
    
    def inverse(self, y, tbe, ten):
        out = self.Physics.inverse(y)
        return out  
    
    def inverse_snr(self, y, snr):
        out = self.Physics.inverse_snr(y, snr)
        return out        
    
    def forward(self, y, invspsf, start, end):
        x = self.InverseNet(y, invspsf, start, end)
        x = x * self.scale
        x = self.Neural(x)
        return x
        
class Net(nn.Module):
    def __init__(self, cfg, type, device='cuda:0'):
        super(Net, self).__init__()
        
        cfg_model = copy.deepcopy(cfg['model'])       
        if type == 'attention':
            self.Neural = unet3d_atte(**cfg_model['Neural']).to(device)
        elif type == 'partial':
            self.Neural = PUNet(**cfg_model['Neural']).to(device)
                    
    def forward(self, x, tbe=None, ten=None):
        out = self.Neural(x, tbe, ten)
        return out
    
class Physics_model(nn.Module):
    def __init__(self, cfg, device='cuda:0'):
        super(Physics_model, self).__init__()
        
        self.Physics = LCT(cfg['physics'], device)
        self.scale_operator = torch.tensor([cfg['model']['scale']]).to(device) 
        self.relu = nn.ReLU(inplace=True)
                    
    def forward_operator(self, x):
        x_compressed = self.Physics.compress(x,  direction=1)  
        y_compressed = self.Physics(x_compressed, self.Physics.fpsf)     
        y_scaled = self.Physics.compress(y_compressed, direction=-1)
        y = self.Physics.scale(y_scaled, direction=-1)
        # y[y<0]=0
        y = self.relu(y)
        y = y*self.scale_operator
        return y

    def inverse_snr(self, y, snr):
        y_scaled = self.Physics.scale(y, direction=1)
        y_compressed = self.Physics.A_compress(y_scaled, direction=1)
        # compute kernel for different snr
        invpsf = torch.conj(self.Physics.fpsf) / (1 / snr + torch.real(self.Physics.fpsf) ** 2 + torch.imag(self.Physics.fpsf) ** 2)
        x_compressed = self.Physics(y_compressed, invpsf)
        x = self.Physics.A_compress(x_compressed, direction=-1)
        # x[x<0]=0
        x = self.relu(x)
        return x
        
    def inverse(self, y, tbe, ten):
        y_scaled = self.Physics.scale(y, direction=1)
        y_compressed = self.Physics.A_compress(y_scaled, direction=1)
        x_compressed = self.Physics(y_compressed, self.Physics.invpsf)
        x = self.Physics.A_compress(x_compressed, direction=-1)
        # x[x<0]=0
        x = self.relu(x)
        return x
     
