import os, glob
import numpy as np
import torch
import matplotlib.image as mpimg
from scipy.io import loadmat, savemat
 
def load_pattern(s_ds, t_ds, fname, t_res):
    for i in range(t_ds):
        t_res = t_res // 2
    
    if os.path.exists(fname):
        mask = mpimg.imread(fname)
        mask = torch.from_numpy(mask)
        mask = mask[None,None,...].repeat(1, t_res, 1, 1)
        return mask
    else:
        raise ValueError("can not find this mask")
    
    
def load_real_data(data_path, t_ds=0):
    transient_name = glob.glob(f"{data_path}/meas.mat")[0]
    with open(glob.glob(f"{data_path}/t_range.txt")[0], "r") as f:
        lines = f.readlines()
        tbe = int(lines[0].strip())
        ten = int(lines[1].strip())
    transient = loadmat(transient_name)['meas'].transpose(2,0,1)[None].astype(np.float32())
    transient = torch.from_numpy(transient)
                                
    # downsample t
    for i in range(t_ds):
        transient = transient[:,::2,...]+transient[:,1::2,...]
        tbe = tbe // 2
        ten = ten // 2
    
    transient = transient/transient.amax(dim=(-3, -2, -1), keepdim=True)*140
    return transient, tbe, ten

    

    








