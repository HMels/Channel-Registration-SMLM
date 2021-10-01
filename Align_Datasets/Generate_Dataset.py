# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:26:49 2021

Random Simulates a dataset

@author: Mels
"""

import numpy as np
import copy
import numpy.random as rnd

from AlignModel import AlignModel
from Align_Datasets.channel_class import channel

class Generate_Dataset(AlignModel):
    def __init__(self, coupled=False, imgshape=[512, 512], deform_on=True, 
                 shift=np.array([20,20]), rotation=0.2, shear=np.array([0.003,0.002]), 
                 scaling=np.array([1.0004,1.0003]), random_deform=False):
        self.imgshape=np.array(imgshape)
        self.coupled=coupled
        self.deform = Deform(deform_on, shift, rotation, shear, scaling, random_deform)
        
        AlignModel.__init__(self)
        
    
    #% generate functions
    def generate_dataset_beads(self, N=216, error=10, noise=0.005):
        # generate channels
        self.ch1 = channel_beads(N, self.imgshape, error, noise)
        self.ch2 = copy.deepcopy(self.ch1)
        
        # Generate localization error
        self.ch1.generate_locerror()
        self.ch2.generate_locerror()
        
        # Generate noise error
        self.ch1.generate_noise()
        self.ch2.generate_noise()
        
        # Copy channel and generate noise
        self.ch2_original=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.center_image()
        
        self.ch1 = channel(self.ch1.imgshape, pos = self.ch1.pos, frame = self.ch1.frame)
        self.ch2 = channel(self.ch2.imgshape, pos = self.ch2.pos, frame = self.ch2.frame)
        self.ch2_original = channel(self.ch2_original.imgshape, pos = self.ch2_original.pos, frame = self.ch2_original.frame)
        self.Nbatch = len(self.ch1)
        
        
    def generate_dataset_clusters(self, Nclust=650, N_per_clust=250, std_clust=7,
                                 error=10, noise=0.005):
        # generate channels
        self.ch1 = channel_clusters(Nclust, std_clust, N_per_clust, self.imgshape,error, noise)
        self.ch2 = copy.deepcopy(self.ch1)
        
        # Generate localization error
        self.ch1.generate_locerror()
        self.ch2.generate_locerror()
        
        # Generate noise error
        self.ch1.generate_noise()
        self.ch2.generate_noise()
        
        # Copy channel and generate noise
        self.ch2_original=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.center_image()
        
        self.ch1 = channel(self.ch1.imgshape, pos = self.ch1.pos, frame = self.ch1.frame)
        self.ch2 = channel(self.ch2.imgshape, pos = self.ch2.pos, frame = self.ch2.frame)
        self.ch2_original = channel(self.ch2_original.imgshape, pos = self.ch2_original.pos, frame = self.ch2_original.frame)        
        self.Nbatch = len(self.ch1)
        
    
    #% functions
    def imgparams(self):
    # calculate borders of system
    # returns a 2x2 matrix containing the edges of the image, a 2-vector containing
    # the size of the image and a 2-vector containing the middle of the image
        img = np.empty([2,2], dtype = float)
        img[0,0] = np.min(( np.min(self.ch1.pos[:,0]), np.min(self.ch2.pos[:,0]) ))
        img[1,0] = np.max(( np.max(self.ch1.pos[:,0]), np.max(self.ch2.pos[:,0]) ))
        img[0,1] = np.min(( np.min(self.ch1.pos[:,1]), np.min(self.ch2.pos[:,1]) ))
        img[1,1] = np.max(( np.max(self.ch1.pos[:,1]), np.max(self.ch2.pos[:,1]) ))
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2    
    
    
    def center_image(self):
        try:
            if self.mid is None:
                self.img, self.imgsize, self.mid = self.imgparams()
        except:
            self.img, self.imgsize, self.mid = self.imgparams()
        
        self.ch1.pos -= self.mid
        self.ch2.pos -= self.mid
        self.ch2_original.pos -= self.mid
        self.img, self.imgsize, self.mid = self.imgparams() 
    
        
        
#%% channel_beads class
class channel_beads:
    def __init__(self, N, imgshape=[512, 512], error=10, noise=.005):
        self.imgshape=np.array(imgshape, dtype=np.float32)
        self.pos = np.float32( self.imgshape * rnd.rand(N,2) - (self.imgshape/2) )
        self.N = N
        self.error = error                                          # localization error
        self.noise = noise                                          # amount of noise
        self.img, self.imgsize, self.mid = self.imgparams()         # image parameters for a single channel
      
    def _xyI(self):
        return np.ones((self.N,1))
    
    #% miscalleneous functions
    def generate_locerror(self):
    # Generates a Gaussian localization error over the localizations
        if self.error != 0:
            self.pos[:,0] += rnd.normal(0, self.error, self.N)
            self.pos[:,1] += rnd.normal(0, self.error, self.N)
            
            
    def generate_noise(self):
    # generates some noise
        if self.noise!=0:
            self.N_noise = int(self.noise * self.N)
            Nlocs = np.array([
                np.float32( self.imgsize[0] * rnd.rand( self.N_noise ) - self.mid[0] ),
                np.float32(self.imgsize[1] * rnd.rand( self.N_noise ) - self.mid[0] )
                ])
            
            self.pos = np.append(self.pos, ( Nlocs.transpose() ), 0)
            self.N = self.pos.shape[0]
        
        
    def imgparams(self):
    # Image parameters for a single channel
        img = np.empty([2,2], dtype = float)
        img[0,0] = np.min(self.pos[:,0])
        img[0,1] = np.max(self.pos[:,0])
        img[1,0] = np.min(self.pos[:,1])
        img[1,1] = np.max(self.pos[:,1])
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
    
    

#%% channel_clusters class
class channel_clusters:
    def __init__(self, Nclust=650, std_clust=7, N_per_clust=250,
                 imgshape=[512, 512], error=10, noise=.005):
        self.imgshape=np.array(imgshape, dtype=np.float32)
        self.clust_locs = np.float32( self.imgshape * rnd.rand(Nclust,2) - (self.imgshape/2))
        self.Nclust = Nclust
        self.std_clust=std_clust
        self.N_per_clust=N_per_clust
        self.error = error                                          # localization error
        self.noise = noise                                          # amount of noise
        
        self.pos = self.generate_cluster_pos()
        self.N = self.pos.shape[0]
        self.img, self.imgsize, self.mid = self.imgparams()         # image parameters for a single channel
        
    def _xyI(self):
        return np.ones((self.N,1))
    
    #% miscalleneous functions
    def generate_locerror(self):
    # Generates a Gaussian localization error over the localizations
        if self.error != 0:
            self.pos[:,0] += rnd.normal(0, self.error, self.N)
            self.pos[:,1] += rnd.normal(0, self.error, self.N)
            
            
    def generate_noise(self):
    # generates some noise
        if self.noise!=0:
            self.N_noise = int(self.noise * self.N)
            Nlocs = np.array([
                np.float32( self.imgsize[0] * rnd.rand( self.N_noise ) - self.mid[0] ),
                np.float32( self.imgsize[1] * rnd.rand( self.N_noise ) - self.mid[0] )
                ])
            
            self.pos = np.append(self.pos, ( Nlocs.transpose() ), 0)
            self.N = self.pos.shape[0]
        
        
    def imgparams(self):
    # Image parameters for a single channel
        img = np.empty([2,2], dtype = float)
        img[0,0] = np.min(self.pos[:,0])
        img[0,1] = np.max(self.pos[:,0])
        img[1,0] = np.min(self.pos[:,1])
        img[1,1] = np.max(self.pos[:,1])
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
    
    
    def generate_cluster_pos(self):
        ## Generating the Cluster Points
        locs = []
        i=0
        while i < self.Nclust:
            sigma = self.std_clust+30*rnd.randn(2)                           # std gets a normal random deviation
            N = int(round(self.N_per_clust*(1+0.5*rnd.randn()),0))    # number of points also 
            if N>0 and sigma[0]>0 and sigma[1]>0:                       # are the points realistic
                locs.append(self.gauss_2d(self.clust_locs[i,:],sigma, N ))
                i+=1
                
        ## Generating more points around the clusters
        i=0
        while i < self.Nclust:
            sigma = 30*(self.std_clust+30*rnd.randn(2))
            N = int(round(self.N_per_clust*(1+0.5*rnd.randn())/5,0))
            if N>0 and sigma[0]>0 and sigma[1]>0:
                locs.append(self.gauss_2d(self.clust_locs[i,:],sigma, N ))
                i+=1
        locs = np.concatenate(locs, axis=0)   # add all points together
        
        ## Fit every point inside image
        locs[:,0] = np.float32( (locs[:,0]+(self.imgshape[0]/2))%self.imgshape[0] - (self.imgshape/2)[0])
        locs[:,1] = np.float32( (locs[:,1]+(self.imgshape[1]/2))%self.imgshape[1] - (self.imgshape/2)[1])
        return locs
        
    
    def gauss_2d(self,mu, sigma, N):
        '''
        Generates a 2D gaussian cluster
        Parameters
        ----------
        mu : 2 float array
            The mean location of the cluster.
        sigma : 2 float array
            The standard deviation of the cluster.
        N : int
            The number of localizations.
        Returns
        -------
        Nx2 float Array
            The [x1,x2] localizations .
        '''
        x1 = np.float32( rnd.normal(mu[0], sigma[0], N) )
        x2 = np.float32( rnd.normal(mu[1], sigma[1], N) )
        return np.array([x1, x2]).transpose()
    
    
    
#%% Deform class
class Deform():
    '''
    This class contains all functions and variables used to give the image a deformation
    
    The variables are:
        - shift
        - rotation
        - shear
        - scaling
    
    The functions are:
        - deformation()
        - ideformation()
        - shift_def()
        - shift_idef()
        - rotation_def()
        - rotation_idef()
        - shear_def()
        - shear_idef()
        - scaling_def()
        - scaling_idef()
    '''
    
    def __init__(self, deform_on=True, shift=None, rotation=None, shear=None, scaling=None,
                 random_deform=False):
        self.deform_on = deform_on
        if random_deform:
            self.shift = np.array([ 20  , 20 ]) + 10*rnd.randn(2)       
            self.rotation = 0.2*rnd.randn(1)
            self.shear=np.array([0,0])#np.array([0.003, 0.002])  + 0.001*rnd.randn(2)
            self.scaling=np.array([1,1])#np.array([1.0004,1.0003 ])+ 0.0001*rnd.randn(2)
        else:
            self.shift = shift if shift is not None else np.array([0.,0.])
            self.rotation = rotation if rotation is not None else 0.
            self.shear = shear if shear is not None else np.array([0.,0.])
            self.scaling = scaling if scaling is not None else np.array([1.,1.])
        
        
    def deform(self, locs):
        if self.deform_on:
            if (self.shift[0] != 0 or self.shift[1] != 0) and self.shift is not None:
                locs = self.shift_def(locs)
            if (self.rotation != 0) and self.rotation is not None:
                locs = self.rotation_def(locs)
            if (self.shear[0] != 0 or self.shear[1] != 0) and self.shear is not None:
                locs = self.shear_def(locs)
            if (self.scaling[0] != 1 or self.scaling[1] != 1) and self.scaling is not None:
                locs = self.scaling_def(locs)
        return locs
    
    
    def ideform(self, locs):
        if self.deform_on:
            if (self.scaling[0] != 1 or self.scaling[1] != 1) and self.scaling is not None:
                locs = self.scaling_idef(locs)
            if (self.shear[0] != 0 or self.shear[1] != 0) and self.shear is not None:
                locs = self.shear_idef(locs)
            if (self.rotation != 0) or self.rotation is not None:
                locs = self.rotation_idef(locs)
            if (self.shift[0] != 0 or self.shift[1] != 0) and self.shift is not None:
                locs = self.shift_idef(locs)
        return locs
        
    
    def shift_def(self, locs):
        return locs + self.shift
    
    
    def shift_idef(self, locs):
        return locs - self.shift
    
    
    def rotation_def(self, locs):
        cos = np.cos(self.rotation * 0.0175) 
        sin = np.sin(self.rotation * 0.0175)
       
        locs = np.array([
             (cos * locs[:,0] - sin * locs[:,1]) ,
             (sin * locs[:,0] + cos * locs[:,1]) 
            ]).transpose()
        return locs
    
    
    def rotation_idef(self, locs):
        cos = np.cos(self.rotation * 0.0175) 
        sin = np.sin(self.rotation * 0.0175)
       
        locs = np.array([
             (cos * locs[:,0] + sin * locs[:,1]) ,
             (-1*sin * locs[:,0] + cos * locs[:,1]) 
            ]).transpose()
        return locs
    
    
    def shear_def(self, locs):
        locs = np.array([
            locs[:,0] + self.shear[0]*locs[:,1] ,
            self.shear[1]*locs[:,0] + locs[:,1] 
            ]).transpose()
        return locs
    
    
    def shear_idef(self, locs):
        locs = np.array([
            locs[:,0] - self.shear[0]*locs[:,1] ,
            -1*self.shear[1]*locs[:,0] + locs[:,1] 
            ]).transpose()
        return locs
    
    
    def scaling_def(self, locs):
        locs = np.array([
            self.scaling[0] * locs[:,0] ,
            self.scaling[1] * locs[:,1]
            ]).transpose()
        return locs
    
    
    def scaling_idef(self, locs):
        locs = np.array([
            (1/self.scaling[0]) * locs[:,0] ,
            (1/self.scaling[1]) * locs[:,1]
            ]).transpose()
        return locs