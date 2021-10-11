# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:54:37 2021

@author: Mels
"""
import copy
import numpy as np
import numpy.random as rnd

from dataset import dataset
from Channel import Channel

class dataset_simulation(dataset):
    def __init__(self, pix_size=1, linked=False, imgshape=[512, 512], 
                 FrameLinking=False, FrameOptimization=False):
        self.pix_size=pix_size    # the multiplicationfactor to change the dataset into units of nm
        self.imgshape=imgshape    # number of pixels of the dataset
        self.linked=linked        # is the data linked/paired?
        self.FrameLinking=FrameLinking              # will the dataset be linked or NN per frame?
        self.FrameOptimization=FrameOptimization    # will the dataset be optimized per frame
        dataset.__init__(self, path=None,pix_size=pix_size,linked=linked,imgshape=imgshape,
                         FrameLinking=FrameLinking,FrameOptimization=FrameOptimization)
        
        
    #%% generate functions
    def generate_dataset_beads(self, N=216, error=10, noise=0.005, deform=None):
        pos1=np.array(self.imgshape*rnd.rand(N,2)*self.pix_size, dtype=np.float32) # generate channel positions
        pos2=copy.copy(pos1)
        pos1=self.generate_locerror(pos1, error) # Generate localization error
        pos2=self.generate_locerror(pos2, error)
        pos1=self.generate_noise(pos1, noise) # Generate noise
        pos2=self.generate_noise(pos2, noise)
        if deform is not None: pos2=deform.deform(pos2) # deform  channel
        #if not self.linked: pos2=self.shuffle(pos2) # if channel is not linked, shuffle indices
        # load into channels
        self.ch1 = Channel(pos=pos1, frame=np.ones(pos1.shape[0]))
        self.ch2 = Channel(pos=pos2, frame=np.ones(pos2.shape[0]))
        # Copy channel and generate noise
        self.ch20=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.center_image()
        
    
    def generate_dataset_clusters(self, Nclust=600, N_per_clust=250, std_clust=25,
                                 error=10, noise=0.1, deform=None):
        pos1 = self.generate_cluster_pos(Nclust, N_per_clust, std_clust) # generate channels
        pos2 = copy.copy(pos1)
        pos1=self.generate_locerror(pos1, error) # Generate localization error
        pos2=self.generate_locerror(pos2, error)
        pos1=self.generate_noise(pos1, noise) # Generate noise
        pos2=self.generate_noise(pos2, noise)        
        if deform is not None: pos2=deform.deform(pos2) # deform  channel
        #if not self.linked: pos2=self.shuffle(pos2) # if channel is not linked, shuffle indices
        # load into channels     
        self.ch1 = Channel(pos=pos1, frame=np.random.choice(np.arange(0,10),(pos1.shape[0])))
        self.ch2 = Channel(pos=pos2, frame=np.random.choice(np.arange(0,10),(pos2.shape[0])))
        # Copy channel and generate noise
        self.ch20=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.center_image() 
               
        
    def generate_cluster_pos(self, Nclust=650, N_per_clust=250, std_clust=7):
        ## Generating the Cluster Points
        clust_locs = np.float32(self.imgshape*rnd.rand(Nclust,2)*self.pix_size)
        
        (pos,i)=([],0)
        while i < Nclust:                                               # iterate over clusters
            sigma = std_clust+10*rnd.randn(2)                           # std gets a normal random deviation
            N = int(round(N_per_clust*(1+0.5*rnd.randn()),0))           # number of points in the cluster 
            if N>0 and sigma[0]>0 and sigma[1]>0:                       # are the points realistic
                pos.append(self.gauss_2d(clust_locs[i,:],sigma, N ))
                i+=1
                
        ## Generating more points around the clusters
        i=0
        while i < Nclust:
            sigma = 10*(std_clust+10*rnd.randn(2))
            N = int(round(N_per_clust*(1+0.5*rnd.randn())/5,0))
            if N>0 and sigma[0]>0 and sigma[1]>0:
                pos.append(self.gauss_2d(clust_locs[i,:],sigma, N ))
                i+=1
        pos = np.concatenate(pos, axis=0)   # add all points together
        
        ## Fit every point inside image
        pos[:,0] = np.float32( pos[:,0]%(self.imgshape[0]*self.pix_size) )
        pos[:,1] = np.float32( pos[:,1]%(self.imgshape[1]*self.pix_size) )
        return pos
        
        
        
    #%% miscalleneous functions
    def generate_locerror(self, pos, error):
    # Generates a Gaussian localization error over the localizations
        self.error=error
        if self.error != 0:
            pos[:,0] += rnd.normal(0, self.error, pos.shape[0])
            pos[:,1] += rnd.normal(0, self.error, pos.shape[0])
        return pos
            
            
    def generate_noise(self, pos, noise):
    # generates some noise
        self.noise=noise
        if self.noise!=0:
            Nlocs = np.array([
                np.float32( self.imgshape[0] * rnd.rand( int(self.noise * pos.shape[0])) * self.pix_size),
                np.float32( self.imgshape[1] * rnd.rand( int(self.noise * pos.shape[0])) * self.pix_size)
                ])
            return np.append(pos, ( Nlocs.transpose() ), axis=0)
        else: return pos
        
        
    def shuffle(self, pos):
        idx=np.arange(0,pos.shape[0]).astype('int')
        rnd.shuffle(idx)
        return pos[idx,:]
        

    def gauss_2d(self, mu, sigma, N):
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
    
    
    def relink_dataset(self):
        self.linked=True 
        N=int(np.min((self.ch1.pos.shape[0],self.ch2.pos.shape[0]))/(1+self.noise))
        frame1=self.ch1.frame.numpy()
        frame2=self.ch2.frame.numpy()
        frame20=self.ch20.frame.numpy()
        pos1=self.ch1.pos.numpy()
        pos2=self.ch2.pos.numpy()
        pos20=self.ch20.pos.numpy()
        del self.ch1, self.ch2, self.ch20
        self.ch1 = Channel( pos1[:N,:] , frame1[:N] )
        self.ch2 = Channel( pos2[:N,:] , frame2[:N] )
        self.ch20 = Channel( pos20[:N,:] , frame20[:N] )
            
            
            
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
    
    def __init__(self, shift=np.array([ 500  , 650 ]), rotation=0.2, 
                 shear=np.array([0.003, 0.002]), scaling=np.array([1.0004,1.0003 ]),
                 random_deform=False):
        self.random_deform=random_deform
        self.shift = shift if shift is not None else np.array([0.,0.])
        self.rotation = rotation if rotation is not None else 0.
        self.shear = shear if shear is not None else np.array([0.,0.])
        self.scaling = scaling if scaling is not None else np.array([1.,1.])
        if random_deform:
            self.shift+=40*rnd.randn(2)       
            self.rotation+=0.2*rnd.randn(1)
            self.shear+=np.array([0,0])#  + 0.001*rnd.randn(2)
            self.scaling+=np.array([0,0])#+ 0.0001*rnd.randn(2)
        
        
    def deform(self, locs):
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