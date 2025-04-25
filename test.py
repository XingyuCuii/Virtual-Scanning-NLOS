import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os, yaml
from libs.load_data import *
from models.worker import *

def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # log/save path
    parser.add_argument('--config', 
                        default='configs/test.yaml',
                        help='config file path')
    parser.add_argument("--dn_pt", type=str, 
                        default='vs_ckp/SURE_denoiser.pth', 
                        help='checkpoint of SURE_denoiser')   
    parser.add_argument("--vs_pt", type=str, 
                        default='vs_ckp/VSNet.pth',
                        help='checkpoint of VSNet') 
    parser.add_argument("--gpu", type=str, default='1', 
                        help='device') 
    parser.add_argument("--snr", type=float, 
                        default=2.e+5, 
                        help='change for different pattern')   
    parser.add_argument('--data_path', 
                        default='test_data/real_data/fk/teaser',
                        help='path of data')       
    parser.add_argument('--pattern_path', 
                        default='test_data/pattern/window1.png',
                        help='path of pattern')       

    return parser

def max_pro(vol):
    output = vol/vol.amax(dim=(-3, -2, -1), keepdim=True)
    MIP = output.max(-3)[0]
    MIP = MIP/MIP.amax(dim=(-3, -2, -1), keepdim=True)
    MIP = torch.clamp(MIP, 0, 1)
    gray_img = MIP[0,0].cpu().numpy()
    return gray_img
    
def main():
    
    parser = config_parser()
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # for sh 
    device='cuda:{}'.format(args.gpu)
    snr = torch.tensor([args.snr]).to(device)
    torch.cuda.set_device(device)
    
    save_path = './output'
    os.makedirs(save_path, exist_ok=True)
    
    ### create model
    SURE_denoiser = VSmodel(cfg, 'partial', device, physics=False)
    VSnet = VSmodel(cfg, 'attention', device)
    
    if args.dn_pt:
        checkpoint = torch.load(args.dn_pt, map_location=device)
        SURE_denoiser.load_state_dict(checkpoint['model_state_dict'])
        print('load SURE_denoiser model dict: {}'.format(args.dn_pt))   
                    
    if args.vs_pt:
        checkpoint = torch.load(args.vs_pt, map_location=device)
        VSnet.load_state_dict(checkpoint['model_state_dict'])
        print('load vs model dict: {}'.format(args.vs_pt))           

    SURE_denoiser.eval()
    VSnet.eval()
    
    # get pattern
    pattern = load_pattern(cfg['physics']['s_ds'], cfg['physics']['t_ds'], args.pattern_path, cfg['physics']['temprol_grid'])
    y_raw, tbe, ten = load_real_data(args.data_path, cfg['physics']['t_ds'])
    
    pattern = pattern[None].to(device)
    y_raw = y_raw[None].to(device)
    tbe = torch.tensor([tbe]).to(device)
    ten = torch.tensor([ten]).to(device)
        
    data_name = args.data_path.split('/')[-1]
    pattern_name = args.pattern_path.split('/')[-1][:-4]
    
    # test
    with torch.no_grad():  
        y_usm = y_raw*pattern
        y_dn = SURE_denoiser.Neural(torch.cat([y_usm, pattern], dim=1), tbe, ten)
        y_dn = y_dn*pattern
                            
        # A+
        x_range = VSnet.inverse_snr(y_dn, snr)
        
        # enhance
        x1 = VSnet.neural(x_range, tbe, ten)

        # save MIP image
        obj_img = max_pro(x1)
        plt.imsave(f'{save_path}/{data_name}_{pattern_name}.png', obj_img, cmap='hot')

if __name__ == '__main__':
    main()
