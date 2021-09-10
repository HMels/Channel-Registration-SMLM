# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:40:56 2021

@author: Mels
"""
import numpy as np
from photonpy import Dataset
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import pandas as pd

from AlignModel import AlignModel

class AlignModelExcel(AlignModel):
    def __init__(self, path, subset=None, align_rcc=True, coupled=False):
        
        self.shift_rcc=None
        self.subset=subset
        self.coupled=coupled
        self.gridsize=None
        self.ch1, self.ch2 = self.load_dataset(path)
        
        
    #%% functions
    def load_dataset(ds, imgshape=[512, 512], shift=None):
        '''
        Loads the dataset
    
        Parameters
        ----------
        ds : str
            The path where the data is located.
    
        Returns
        -------
        locs_A , locs_B : : Nx2 nupy array
            Localizations of both channels.
    
        '''
        data = pd.read_csv(ds)
        grouped = data.groupby(data.Channel)
        ch1 = grouped.get_group(1)
        ch2 = grouped.get_group(2)
        
        data1 = np.array(ch1[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data1 = np.column_stack((data1, np.arange(data1.shape[0])))
        data2 = np.array(ch2[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data2 = np.column_stack((data2, np.arange(data2.shape[0])))
    
        locs1 = rcc.Localizations(data1, imgshape)
        locs2 = rcc.Localizations(data2, imgshape)
        if shift is None:
            shift=locs1.align(locs2)
            print('Shifted with RCC of', shift)
            
        locs1.pos += shift

        ch1 = Dataset(length=data.shape[0], dims=2, imgshape=imgshape,)
            
        return ch1, ch2
    
    
#%%
class Localizations:
    def __init__(self, data, imgshape):
        self.pos = data[:,:2]
        self.frame = data[:,2]
        self._xyI = data[:,3]
        self.index = data[:,4]
        self.imgshape = np.array(imgshape)
        self.len = data.shape[0]       #### obviously false but it works for now
        # The indices of the coupled datasets
        self.NNidx = None
        
        
    def compile_data(self):
        return np.concatenate((self.pos, self.frame[:,None],self._xyI[:,None], self.index[:,None]), axis=1)
    
    def copy_channel(self):
        return Localizations(self.compile_data(), self.imgshape)
    
    def generate_xyI(self):
    # creates an intensity of 1 for each pos
        r=np.zeros((self.len,3))
        r[:,:2] = self.pos[:,:2]
        r[:,2] = 1
        return r
    
    def align(self, other):
        xyI = np.concatenate([self.generate_xyI(), other.generate_xyI()])
        framenum = np.concatenate([np.zeros(self.len,dtype=np.int32), np.ones(other.len,dtype=np.int32)])
        return 2*rcc(xyI, framenum, 2, np.max(self.imgshape), maxdrift=10,zoom=2,RCC=False)[0][1]
    
    
    def renderGaussianSpots(self, zoom=1, sigma=1):
        imgshape = np.array(self.imgshape)*zoom
        with Context() as ctx:
            img = np.zeros(imgshape,dtype=np.float64)
            spots = np.zeros((self.len, 5), dtype=np.float32)
            spots[:, 0] = self.pos[:,0] * zoom
            spots[:, 1] = self.pos[:,1] * zoom
            spots[:, 2] = .15
            spots[:, 3] = .15
            spots[:, 4] = 1
            return GaussianPSFMethods(ctx).Draw(img, spots)
        
    
    def couple_dataset(self, other, maxDist=50, Filter=True):
        print('Coupling datasets with an iterative method...')
        
        self.NNidx = []
        other.NNidx = []
        Dist = []
        for i in range(self.len):
            # First find the positions in the same frame
            sameframe_pos = np.squeeze(other.pos[np.argwhere(other.frame==self.frame[i]),:], axis=1)
            sameframe_idx = np.squeeze(other.index[np.argwhere(other.frame==self.frame[i])], axis=1)
            
            dists = np.sqrt(np.sum((self.pos[i,:]-sameframe_pos)**2,1))
            if not Filter or np.min(dists)<maxDist: 
                locA=self.pos[i,:]
                locB=sameframe_pos[np.argmin(dists),:]
                
                Dist.append(locA-locB)
                self.NNidx.append(i)
                other.NNidx.append( sameframe_idx[np.argmin(dists)].astype('int') )                
        return self.coupled_data(), other.coupled_data(), np.array(Dist)
    
    
    def coupled_data(self):
        if self.NNidx is None:
            print('Error: Dataset not yet coupled. Use couple_dataset()')
        elif self.NNidx == []:
            print('Error: No data has been coupled')
        else:
            return self.pos[self.NNidx,:]    
        
    
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