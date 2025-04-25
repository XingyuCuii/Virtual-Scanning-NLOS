import torch, os
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d

class LCT(nn.Module):
    
    def __init__(self, cfg, device):
        super(LCT, self).__init__()
        
        self.spatial_grid = cfg['spatial_grid']
        self.temprol_grid = cfg['temprol_grid']
        self.wall_size = cfg['wall_size']
        self.c = cfg['c']
        self.bin_len = cfg['bin_len']
        
        for i in range(cfg['s_ds']):
            self.spatial_grid = int(self.spatial_grid / 2)
        
        for i in range(cfg['t_ds']):
            self.temprol_grid = self.temprol_grid  // 2
            self.bin_len = self.bin_len * 2
            
        self.device = device
        self.method = cfg['method']
        self.material = cfg['material']
        assert 2 ** int(np.log2(self.temprol_grid)) == self.temprol_grid
        
        ########################################################3
        trange = self.temprol_grid * self.bin_len/2 
        slope = self.wall_size / 2 /  trange
        self.psf_init(slope)
        
        gridz_M = np.linspace(0, 1/slope, self.temprol_grid, dtype=np.float32)
        gridz_1xMx1x1 = gridz_M.reshape(1,-1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32)).to(self.device)
        
        sample_path = 'models/physics/kernel/RTZ_{}_{:.4f}.pt'.format(self.temprol_grid,slope)
        if os.path.exists(sample_path):
            RTZ = torch.load(sample_path, map_location= self.device)
            RT = RTZ['RT']
            RZ = RTZ['RZ']
        else:
            RTZ = {}
            RT, RZ = self.resamplingOperator(self.temprol_grid, start=0, stop=1/slope)
            RTZ['RT'] = RT
            RTZ['RZ'] = RZ
            torch.save(RTZ, sample_path)
        
        self.RT_MxM = RT.to(self.device)   
        self.RZ_MxM = RZ.to(self.device)
        self.filter = torch.from_numpy(self.filterLaplacian()).to(self.device)
    
    def scale(self, data_bxdxtxwxh, direction):
        '''
        direction: forward=-1 backward=1
        '''
        bnum, dnum, tnum, wnum, hnum = data_bxdxtxwxh.shape        
        data_BDxTxWxH = data_bxdxtxwxh.reshape(bnum*dnum, tnum, hnum, wnum)

        if self.material == 'diffuse':
                data_BDxTxWxH = data_BDxTxWxH * ((self.gridz_1xMx1x1) ** 4) ** direction
        elif self.material == 'specular':
                data_BDxTxWxH = data_BDxTxWxH * ((self.gridz_1xMx1x1) ** 2) ** direction

        return data_BDxTxWxH.reshape(bnum, dnum, tnum, hnum, wnum)

    def compress(self, data_bxdxtxwxh, direction):
        '''
        direction: decompressed=-1 compress=1
        '''
        bnum, dnum, tnum, wnum, hnum = data_bxdxtxwxh.shape        
        data_BDxTxWxH = data_bxdxtxwxh.reshape(bnum*dnum, tnum, hnum, wnum)

        if direction==1:
            left  = self.RT_MxM
        else:
            left = self.RZ_MxM

        right = data_BDxTxWxH.reshape(bnum*dnum, tnum, -1)
        tdata = torch.matmul(left, right)
       
        output = tdata.reshape(bnum, dnum, self.temprol_grid, hnum, wnum)
                        
        return output
    
    def A_compress(self, data_bxdxtxwxh, direction):
        '''
        direction: decompressed=-1 compress=1
        '''
        bnum, dnum, tnum, wnum, hnum = data_bxdxtxwxh.shape        
        data_BDxTxWxH = data_bxdxtxwxh.reshape(bnum*dnum, tnum, hnum, wnum)

        if direction==1:
            left = self.RT_MxM
        else:
            left = self.RZ_MxM

        right = data_BDxTxWxH.reshape(bnum*dnum, tnum, -1)
        tdata = torch.matmul(left, right)
        
        output = tdata.reshape(bnum, dnum, self.temprol_grid, hnum, wnum)
                        
        return output
    
    
    def psf_init(self, slope):
        os.makedirs('models/physics/kernel', exist_ok=True)
        psf_path = 'models/physics/kernel/psf_{}_{}_{:.4f}.pt'.format(self.temprol_grid,self.spatial_grid,slope)
        if os.path.exists(psf_path):
            psf = torch.load(psf_path, map_location= self.device)
        else:
            psf = torch.from_numpy(self.definePsf(self.spatial_grid, self.temprol_grid, slope))
            torch.save(psf, psf_path)
           
        fpsf = torch.fft.fftn(psf)
        self.fpsf = fpsf[None,...].to(self.device)
 
    def forward(self, data, kernel):
        
        # 1 padd data with zero
        bnum, dnum, tnum, wnum, hnum = data.shape
         
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = tnum
        
        # ####################################################
        # # 3 run lct
        datapad_BDx2Tx2Wx2H = torch.zeros((bnum*dnum, tnum*2, hnum*2, wnum*2), dtype=torch.float32).to(data)
        datapad_BDx2Tx2Wx2H[:, :temprol_grid, :sptial_grid, :sptial_grid] = data.reshape(bnum*dnum, temprol_grid, sptial_grid, sptial_grid)
        
        # Step 3: Convolve with inverse filter and unpad result
        fre_datapad_BDx2Tx2Wx2H = torch.fft.fftn(datapad_BDx2Tx2Wx2H, dim=(-3,-2,-1)) 
        fitered_datapad_BDx2Tx2Wx2H = fre_datapad_BDx2Tx2Wx2H * kernel 
        vol_complex = torch.fft.ifftn(fitered_datapad_BDx2Tx2Wx2H, dim=(-3,-2,-1)).real.float() 
        volumn_BDxTxWxH = vol_complex[:, :temprol_grid, :sptial_grid, :sptial_grid] 
        
        #  Step 4: Resample depth axis and clamp results
        right = volumn_BDxTxWxH.reshape(bnum, dnum, tnum, hnum, wnum)
        
        return right
    
    def todev(self, dev):
        self.gridz_1xMx1x1_todev = self.gridz_1xMx1x1.to(dev)
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
    
    def definePsf(self, U, V, slope):
                                 
        x = np.linspace(-1, 1, 2*U, dtype=np.float32)
        y = np.linspace(-1, 1, 2*U, dtype=np.float32)
        z = np.linspace(0, 2, 2*V, dtype=np.float32)

        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')

        # Define PSF
        psf = np.abs((4*slope**2) * ((grid_x)**2 + (grid_y)**2) - grid_z)
        min_psf = np.min(psf, axis=0)
        psf = (psf == np.tile(min_psf, (2*V, 1, 1))).astype(float)
        psf = psf / np.sum(psf)
        psf = np.roll(psf, (0, U, U), axis=(0, 1, 2))

        return psf

    def resamplingOperator(self, M, start, stop):
                
        x = np.linspace(start, stop, M, dtype=np.float64)**2
        val = np.eye(M,M)
        xq = np.linspace(start**2, stop**2, M**2, dtype=np.float64)
        
        interp_func = interp1d(x, val, kind='linear', axis=0, fill_value='extrapolate')
        mtx = interp_func(xq)
        
        K = int(np.log2(M))
        for _ in np.arange(K):
            mtx = 0.5 * (mtx[0::2, :] + mtx[1::2])
            
        mtx = mtx / np.tile(np.sum(mtx, axis=0), (mtx.shape[0], 1))
        mtx[np.isnan(mtx)] = 0

        mtxi = mtx.T
        mtxi = mtxi / np.tile(np.sum(mtxi, axis=0), (mtxi.shape[0], 1))
        mtxi[np.isnan(mtxi)] = 0

        return torch.from_numpy(mtx.astype(np.float32)), torch.from_numpy(mtxi.astype(np.float32))

    def filterLaplacian(self):
        
        hszie = 5
        std1 = 1.0
        
        lim = (hszie - 1) // 2
        std2 = std1 ** 2
        
        dims = np.arange(-lim, lim + 1, dtype=np.float32)
        [y, x, z] = np.meshgrid(dims, dims, dims)
        w = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * std2))
        w = w / np.sum(w)
        
        w1 = w * (x ** 2 + y ** 2 + z ** 2 - 3 * std2)
        w1 = w1 / (std2 ** 2)
        w = w1 - np.mean(w1)
        
        return w
    
