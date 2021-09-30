# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:38:03 2021

@author: Mels
"""
import numpy as np
import tqdm
from photonpy import GaussianPSFMethods
from photonpy.cpp.context import Context
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from photonpy.gaussian.fitters import fit_sigma_2d
from photonpy.cpp.lib import SMLM


#%% channel 
class channel:
    def __init__(self, imgshape=[512, 512], pos=None, frame=None, _xyI=None,index=None):
        self.pos = pos if pos is not None else {}
        self.frame = frame if frame is not None else {}
        #self._xyI = _xyI if _xyI is not None else {}
        self.index = index if index is not None else {}
        self.N = pos.shape[0] if pos is not None else 0
        self.imgshape=imgshape 
        
        
    def compile_data(self):
        return np.concatenate((self.pos, self.frame[:,None],self._xyI[:,None], self.index[:,None]), axis=1)
        
    def _xyI(self):
    # creates an intensity of 1 for each pos
        r=np.zeros((self.N,3))
        r[:,:2] = self.pos[:,:2]
        r[:,2] = 1
        return r
    
    def align(self, other):
        xyI = np.concatenate([self.generate_xyI(), other.generate_xyI()])
        framenum = np.concatenate([np.zeros(self.N,dtype=np.int32), np.ones(other.N,dtype=np.int32)])
        return 2*rcc(xyI, framenum, 2, np.max(self.imgshape), maxdrift=10,zoom=2,RCC=False)[0][1]
    
    
    def renderGaussianSpots(self, zoom=1, sigma=1):
        imgshape = np.array(self.imgshape)*zoom
        with Context() as ctx:
            img = np.zeros(imgshape,dtype=np.float32)
            spots = np.zeros((self.N, 5), dtype=np.float32)
            spots[:, 0] = self.pos[:,0] * zoom
            spots[:, 1] = self.pos[:,1] * zoom
            spots[:, 2] = .15
            spots[:, 3] = .15
            spots[:, 4] = 1
            return GaussianPSFMethods(ctx).Draw(img, spots)
        
    
def rcc(xyI, framenum, timebins, rendersize, maxdrift=3, wrapfov=1, zoom=1, 
        sigma=1, maxpairs=1000,RCC=True,smlm:SMLM=None,useCuda=False):
#    area = np.ceil(np.max(xyI[:,[0,1]],0)).astype(int)
 #   area = np.array([area[0],area[0]])
    
    area = np.array([rendersize,rendersize])
    
    nframes = np.max(framenum)+1
    framesperbin = nframes/timebins
        
    with Context(smlm) as ctx:
        g = GaussianPSFMethods(ctx)
        
        imgshape = area*zoom//wrapfov
        images = np.zeros((timebins, *imgshape))
                    
        for k in range(timebins):
            img = np.zeros(imgshape,dtype=np.float32)
            
            indices = np.nonzero((0.5 + framenum/framesperbin).astype(int)==k)[0]

            spots = np.zeros((len(indices), 5), dtype=np.float32)
            spots[:, 0] = (xyI[indices,0] * zoom) % imgshape[1]
            spots[:, 1] = (xyI[indices,1] * zoom) % imgshape[0]
            spots[:, 2] = sigma
            spots[:, 3] = sigma
            spots[:, 4] = xyI[indices,2]
            
            if len(spots) == 0:
                raise ValueError(f'no spots in bin {k}')

            images[k] = g.Draw(img, spots)

        #print(f"RCC pairs: {timebins*(timebins-1)//2}. Bins={timebins}")
        if RCC:
            pairs = np.array(np.triu_indices(timebins,1)).T
            if len(pairs)>maxpairs:
                pairs = pairs[np.random.choice(len(pairs),maxpairs)]
            pair_shifts = findshift_pairs(images, pairs, ctx.smlm, useCuda=useCuda)
            
            A = np.zeros((len(pairs),timebins))
            A[np.arange(len(pairs)),pairs[:,0]] = 1
            A[np.arange(len(pairs)),pairs[:,1]] = -1
            
            inv = np.linalg.pinv(A)
            shift_x = inv @ pair_shifts[:,0]
            shift_y = inv @ pair_shifts[:,1]
            shift_y -= shift_y[0]
            shift_x -= shift_x[0]
            shift = -np.vstack((shift_x,shift_y)).T / zoom
        else:
            pairs = np.vstack((np.arange(timebins-1)*0,np.arange(timebins-1)+1)).T
            shift = np.zeros((timebins,2))
            shift[1:] = findshift_pairs(images, pairs, ctx.smlm)
            shift /= zoom
            #shift = np.cumsum(shift,0)
            
        t = (0.5+np.arange(timebins))*framesperbin
        
        shift -= np.mean(shift,0)

        shift_estim = np.zeros((len(shift),3))
        shift_estim[:,[0,1]] = shift
        shift_estim[:,2] = t

        if timebins != nframes:
            spl_x = InterpolatedUnivariateSpline(t, shift[:,0], k=2)
            spl_y = InterpolatedUnivariateSpline(t, shift[:,1], k=2)
        
            shift_interp = np.zeros((nframes,2))
            shift_interp[:,0] = spl_x(np.arange(nframes))
            shift_interp[:,1] = spl_y(np.arange(nframes))
        else:
            shift_interp = shift
                
            
    return shift_interp, shift_estim, images


def findshift_pairs(images, pairs, smlm:SMLM, useCuda=True):
    fft2 = smlm.FFT2 if useCuda else np.fft.fft2
    ifft2 = smlm.IFFT2 if useCuda else np.fft.ifft2
    
    print(f"RCC: Computing image cross correlations. Cuda={useCuda}. Image stack shape: {images.shape}. Size: {images.size*4//1024//1024} MB",flush=True)
    w = images.shape[-1]
    if False:
        fft_images = fft2(images)
        fft_conv = np.zeros((len(pairs), w, w),dtype=np.complex64)
        for i, (a,b) in enumerate(pairs):
            fft_conv[i] = np.conj(fft_images[a]) * fft_images[b]
            
        cc =  ifft2(fft_conv)
        cc = np.abs(np.fft.fftshift(cc, (-2, -1)))

        shift = np.zeros((len(pairs),2))
        for i in tqdm.trange(len(pairs)):
            shift[i] = findshift(cc[i], smlm)
    else:
        fft_images = np.fft.fft2(images)
        shift = np.zeros((len(pairs),2))
        # low memory use version
        for i, (a,b) in tqdm.tqdm(enumerate(pairs),total=len(pairs)):
            fft_conv = np.conj(fft_images[a]) * fft_images[b]
            
            cc =  np.fft.ifft2(fft_conv)
            cc = np.abs(np.fft.fftshift(cc))
            shift[i] = findshift(cc, smlm)
    
    return shift


def findshift(cc, smlm:SMLM, plot=False):
    # look for the peak in a small subsection
    r = 6
    hw = 20
    cc_middle = cc[cc.shape[0] // 2 - hw : cc.shape[0] // 2 + hw, cc.shape[1] // 2 - hw : cc.shape[1] // 2 + hw]
    peak = np.array(np.unravel_index(np.argmax(cc_middle), cc_middle.shape))
    peak += [cc.shape[0] // 2 - hw, cc.shape[1] // 2 - hw]
    
    peak = np.clip(peak, r, np.array(cc.shape) - r)
    roi = cc[peak[0] - r + 1 : peak[0] + r, peak[1] - r + 1 : peak[1] + r]
    if plot:
        plt.figure()
        plt.imshow(cc_middle)
        plt.figure()
        plt.imshow(roi)

    #with Context(smlm) as ctx:
        #psf=GaussianPSFMethods(ctx).CreatePSF_XYIBgSigma(len(roi), 1, False)
        #e = psf.Estimate([roi])
        #print(e[0])
#    px,py=phasor_localize(roi)
        #px,py = e[0][0][:2]
    px,py = fit_sigma_2d(roi, initial_sigma=2)[[0, 1]]
    #            roi_top = lsqfit.lsqfitmax(roi)
    return (peak[1] + px - r + 1 - cc.shape[1] / 2), (peak[0] + py - r + 1 - cc.shape[0] / 2)