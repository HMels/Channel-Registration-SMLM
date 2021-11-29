# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:54:37 2021

@author: Mels
"""
import copy
import numpy as np
import numpy.random as rnd
import pandas as pd

from photonpy import Dataset
from dataset import dataset
from Channel import Channel

class dataset_simulation(dataset):
    def __init__(self, pix_size=1, linked=False, loc_error=10, imgshape=[512, 512], 
                 FrameLinking=False, BatchOptimization=False):
        self.pix_size=pix_size    # the multiplicationfactor to change the dataset into units of nm
        self.imgshape=imgshape    # number of pixels of the dataset
        self.loc_error=loc_error  # localization error
        self.linked=linked        # is the data linked/paired?
        self.linked_original=linked
        self.FrameLinking=FrameLinking              # will the dataset be linked or NN per frame?
        self.BatchOptimization=BatchOptimization    # will the dataset be optimized per frame
        dataset.__init__(self, path=None,pix_size=pix_size,linked=linked,imgshape=imgshape,
                         FrameLinking=FrameLinking, loc_error=loc_error, BatchOptimization=BatchOptimization)
        
        
    #%% generate functions
    def generate_dataset_grid(self, N=216, deform=None):
        x = np.linspace(-self.imgshape[0]/2*self.pix_size, self.imgshape[0]/2*self.pix_size, int(np.sqrt(N)))
        y = np.linspace(-self.imgshape[1]/2*self.pix_size, self.imgshape[1]/2*self.pix_size, int(np.sqrt(N)))
        pos1=np.reshape(np.stack(np.meshgrid(x,y), axis=2),[-1,2])
        pos2=copy.copy(pos1)
        pos1=self.generate_locerror(pos1, self.loc_error) # Generate localization error
        pos2=self.generate_locerror(pos2, self.loc_error)
        if deform is not None: pos2=deform.deform(pos2) # deform  channel
        # load into channels
        self.ch1 = Channel(pos=pos1, frame=np.ones(pos1.shape[0]))
        self.ch2 = Channel(pos=pos2, frame=np.ones(pos2.shape[0]))
        # Copy channel
        self.ch20=copy.deepcopy(self.ch2)
        self.ch20linked=copy.deepcopy(self.ch2)
        self.ch10=copy.deepcopy(self.ch1)
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.center_image()
        
        
    def generate_dataset_beads(self, N=216, deform=None):
        pos1=np.array(self.imgshape*rnd.rand(N,2)*self.pix_size, dtype=np.float32) # generate channel positions
        pos2=copy.copy(pos1)
        pos1=self.generate_locerror(pos1, self.loc_error) # Generate localization error
        pos2=self.generate_locerror(pos2, self.loc_error)
        if deform is not None: pos2=deform.deform(pos2) # deform  channel
        #if not self.linked: pos2=self.shuffle(pos2) # if channel is not linked, shuffle indices
        # load into channels
        self.ch1 = Channel(pos=pos1, frame=np.ones(pos1.shape[0]))
        self.ch2 = Channel(pos=pos2, frame=np.ones(pos2.shape[0]))
        # Copy channel
        self.ch20=copy.deepcopy(self.ch2)
        self.ch20linked=copy.deepcopy(self.ch2)
        self.ch10=copy.deepcopy(self.ch1)
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.center_image()
        
    
    def generate_dataset_clusters(self, Nclust=600, N_per_clust=250, std_clust=25,
                                 error=10, deform=None):
        pos1 = self.generate_cluster_pos(Nclust, N_per_clust, std_clust) # generate channels
        pos2 = copy.copy(pos1)
        pos1=self.generate_locerror(pos1, error) # Generate localization error
        pos2=self.generate_locerror(pos2, error)     
        if deform is not None: pos2=deform.deform(pos2) # deform  channel
        #if not self.linked: pos2=self.shuffle(pos2) # if channel is not linked, shuffle indices
        # load into channels     
        self.ch1 = Channel(pos=pos1, frame=np.random.choice(np.arange(0,10),(pos1.shape[0])))
        self.ch2 = Channel(pos=pos2, frame=np.random.choice(np.arange(0,10),(pos2.shape[0])))
        # Copy channel
        self.ch20=copy.deepcopy(self.ch2)
        self.ch20linked=copy.deepcopy(self.ch2)
        self.ch10=copy.deepcopy(self.ch1)
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
        self.loc_error=error
        if self.loc_error != 0:
            pos[:,0] += rnd.normal(0, self.loc_error, pos.shape[0])
            pos[:,1] += rnd.normal(0, self.loc_error, pos.shape[0])
        return pos
        
        
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
         
            
#%% Dataset copy
class dataset_copy(dataset):
    def __init__(self, path, pix_size=1, loc_error=10, linked=False, imgshape=[512, 512], 
                 FrameLinking=True, BatchOptimization=False):
        self.path=path            # the string or list containing the strings of the file location of the dataset
        self.pix_size=pix_size    # the multiplicationfactor to change the dataset into units of nm
        self.loc_error=loc_error  # localization error
        self.imgshape=imgshape    # number of pixels of the dataset
        self.linked=linked        # is the data linked/paired?
        self.FrameLinking=FrameLinking              # will the dataset be linked or NN per frame?
        self.BatchOptimization=BatchOptimization    # will the dataset be optimized per frame
        self.subset=1
        dataset.__init__(self, path,pix_size=pix_size,linked=linked,imgshape=imgshape,
                         FrameLinking=FrameLinking,BatchOptimization=BatchOptimization)
    
    
    def load_copydataset_hdf5(self, deform):
        print('Loading dataset...')
        ds = Dataset.load(self.path,saveGroups=True)
        try: ch1 = ds[ds.group==0]
        except: ch1 = ds
        
        pos1 = ch1.pos* self.pix_size
        pos2 = copy.copy(pos1)
        pos1 = self.generate_locerror(pos1, self.loc_error) # Generate localization error
        pos2 = self.generate_locerror(pos2, self.loc_error) # Generate localization error
        pos2 = deform.deform(pos2) # deform  channel
        self.ch1 = Channel(pos = pos1, frame = ch1.frame)
        self.ch2 = Channel(pos = pos2, frame = ch1.frame)
        
        self.ch20=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams()                     # loading the image parameters
        self.center_image()
        
        
    #%% miscalleneous functions
    def generate_locerror(self, pos, error):
    # Generates a Gaussian localization error over the localizations
        self.loc_error=error
        if self.loc_error != 0:
            pos[:,0] += rnd.normal(0, self.loc_error, pos.shape[0])
            pos[:,1] += rnd.normal(0, self.loc_error, pos.shape[0])
        return pos
        
        
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
    
    
#%% Affine Deform
class Affine_Deform():
    def __init__(self,A=np.array([[ 1.0031357 ,  0.00181658, -1.3986971], 
                                  [-0.00123012,  0.9972918, 3.3556707 ]]) ):
        self.A=A if A is not None else np.array([ [1,0,0],[0,1,0] ])
        
    def deform(self, locs):
        x1 = locs[:,0]*self.A[0,0] + locs[:,1]*self.A[0,1]
        x2 = locs[:,0]*self.A[1,0] + locs[:,1]*self.A[1,1]
        x1+=self.A[0,2]
        x2+=self.A[1,2]
        return np.stack([x1, x2], axis =1 )
        
    def ideform(self, locs):
        locs-=self.A[:,2]
        det=self.A[0,0]*self.A[1,1]-self.A[1,0]*self.A[0,1]
        if det==0: raise ValueError('Affine transform is not invertible')
        x1 = locs[:,0]*self.A[1,1] - locs[:,1]*self.A[0,1]
        x2 = -locs[:,0]*self.A[1,0] + locs[:,1]*self.A[0,0]
        return np.stack([x1, x2], axis =1 )/det
    
    
    
#%% Examples of loading simulation datasets
if False: #% copy clusters
    DS1 = dataset_copy('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5',
                  linked=False, pix_size=159, loc_error=10, FrameLinking=False, BatchOptimization=True)
    deform=Affine_Deform()
    #deform=Deform(random_deform=False, shift=None ) #,shear=None, scaling=None)
    DS1.load_copydataset_hdf5(deform)
    #DS1 = DS1.SubsetRandom(subset=0.2, linked=True)
    DS1 = DS1.SubsetWindow(subset=0.2, linked=True)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    DS1.find_neighbours(maxDistance=1000)
    
    ## optimization params
    learning_rates = [1000, .1, 1e-4]
    epochs = [5, None, 10]
    pair_filter = [2000, 2000, 2000]
    gridsize=3000
    

if False: #% generate dataset beads
    DS1 = dataset_simulation(imgshape=[256, 512], loc_error=1.4, linked=True,
                             pix_size=159, FrameLinking=False, BatchOptimization=False)
    deform=Deform(shear=None, scaling=None, random_deform=False)
    DS1.generate_dataset_beads(N=216, deform=deform)
    #DS1, DS2 = DS1.SplitDataset(linked=True)
    DS2=None
    
    ## optimization params
    learning_rates = [1000, 1, 1e-2]
    epochs = [100, 500, 300]
    pair_filter = [1000, 10, 10]
    gridsize=500
    
    
if False: #% generate dataset clusters
    DS1 = dataset_simulation(imgshape=[256, 512], loc_error=10, linked=False, 
                             pix_size=159, FrameLinking=False, BatchOptimization=True)
    deform=Deform(random_deform=False, shift=None ) #,shear=None, scaling=None)
    DS1.generate_dataset_clusters(deform=deform)
    #DS1 = DS1.SubsetRandom(subset=0.2, linked=True)
    DS1 = DS1.SubsetWindow(subset=0.5, linked=True)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    DS1.find_neighbours(maxDistance=1000)
    
    ## optimization params
    learning_rates = [1000, 1, 1e-5]
    epochs = [5, 20, 10]
    pair_filter = [800, 400, 400]
    gridsize=1000