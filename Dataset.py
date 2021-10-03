# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:17:52 2021

@author: Mels
"""
import copy
import pandas as pd
import tensorflow as tf
import numpy as np
from photonpy import PostProcessMethods, Context

from Channel import Channel        


#%% Dataset
class Dataset:#(AlignModel):
    def __init__(self, path, pix_size=1, linked=False, imgshape=[512, 512]):
        self.path=path
        self.pix_size=pix_size
        self.imgshape=imgshape
        self.linked=linked        
        #AlignModel.__init__(self)
        
        
    #%% load_dataset
    def load_dataset_excel(self):
        data = pd.read_csv(self.path)
        grouped = data.groupby(data.Channel)
        ch1 = grouped.get_group(1)
        ch2 = grouped.get_group(2)
        
        data1 = np.array(ch1[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data1 = np.column_stack((data1, np.arange(data1.shape[0])))
        data2 = np.array(ch2[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data2 = np.column_stack((data2, np.arange(data2.shape[0])))
    
        self.ch1 = Channel(pos = data1[:,:2], frame = data1[:,2])
        self.ch2 = Channel(pos = data2[:,:2], frame = data2[:,2])
        self.ch1.pos[0].assign(self.ch1.pos[0] * self.pix_size)
        self.ch2.pos[0].assign(self.ch2.pos[0] * self.pix_size)
        
        self.ch2_original=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams()                     # loading the image parameters
        self.center_image()
    
    
    def load_dataset_hdf5(self):
        data = pd.read_csv(self.path)
        grouped = data.groupby(data.Channel)
        ch1 = grouped.get_group(1)
        ch2 = grouped.get_group(2)
        
        data1 = np.array(ch1[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data1 = np.column_stack((data1, np.arange(data1.shape[0])))
        data2 = np.array(ch2[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data2 = np.column_stack((data2, np.arange(data2.shape[0])))
    
        self.ch1 = Channel(pos = data1[:,:2], frame = data1[:,2])
        self.ch2 = Channel(pos = data2[:,:2], frame = data2[:,2])
        self.ch1.pos[0].assign(self.ch1.pos[0] * self.pix_size)
        self.ch2.pos[0].assign(self.ch2.pos[0] * self.pix_size)
        
        self.ch2_original=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams()                     # loading the image parameters
        self.center_image()
        

    #%% functions    
    def imgparams(self):
    # calculate borders of system
    # returns a 2x2 matrix containing the edges of the image, a 2-vector containing
    # the size of the image and a 2-vector containing the middle of the image
        img1, _, _ = self.ch1.imgparams()
        img2, _, _ = self.ch2.imgparams()
        
        img = tf.Variable([[ tf.reduce_min([img1[0,0], img2[0,0]]), tf.reduce_min([img1[0,1], img2[0,1]]) ],
                           [ tf.reduce_max([img1[1,0], img2[1,0]]),  tf.reduce_max([img1[1,1], img2[1,1]]) ]], dtype=tf.float32)

        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
    
    
    def center_image(self):
        self.img, self.imgsize, self.mid = self.imgparams() 
        for batch in range(len(self.ch1.pos)):
            self.ch1.pos[batch] = self.ch1.pos[batch] - self.mid[None,:]
            self.ch2.pos[batch] = self.ch2.pos[batch] - self.mid[None,:]
            self.ch2_original.pos[batch] = self.ch2_original.pos[batch] - self.mid[None,:]
        self.img, self.imgsize, self.mid = self.imgparams() 
        
        
    #%% pair_functions
    #@tf.function
    def link_dataset(self, FrameLinking=False):
    # links dataset with a simple iterative nearest neighbour method
    # FrameLinking links the dataset per frame
        print('Coupling datasets with an iterative method...')
        if len(self.ch1.pos)>1: raise Exception('Dataset should be linked before splitting!')
        
        (locsB,frameB)=([],[])
        for i in range(self.ch1.pos[0].shape[0]):
            if FrameLinking:
                sameframe_idx = tf.where(self.ch2.frame[0]==self.ch1.frame[0][i])
                sameframe_pos = tf.gather(self.ch2.pos[0], sameframe_idx, axis=0)
                dists = tf.sqrt(tf.reduce_sum((self.ch1.pos[0][i,:]-sameframe_pos)**2,axis=1))
            
                iB=tf.argmin(dists)
                locsB.append(tf.gather(sameframe_pos, iB, axis=0))
                frameB.append(self.ch1.frame[0][i])
            else:
                dists = tf.sqrt(tf.reduce_sum((self.ch1.pos[0][i,:]-self.ch2.pos[0])**2,axis=1))
                iB=tf.argmin(dists)
                print(dists.shape, iB)
                locsB.append(tf.gather(self.ch2.pos[0], iB, axis=0))
                frameB.append(self.ch2.frame[0][iB])
            
            self.ch2.pos[0] = tf.Variable(locsB)
            self.ch2.frame[0] = tf.Variable(frameB)
            
        self.linked = True
        
        
    def Filter_Pairs(self, maxDist=150):
    # Filter pairs above maxDist
        print('Filtering pairs above',maxDist,'nm...')
        if not self.linked: raise Exception('Dataset should be linked before filtering pairs')
        
        for batch in range(len(self.ch1.pos)):
            dists = tf.sqrt(tf.reduce_sum( (self.ch1.pos[batch] - self.ch2.pos[batch])**2 ,axis=1))
            idx = tf.where(dists<maxDist)
            
            if idx.shape[0]==0: raise ValueError('All localizations will be filtered out in current settings.')
            self.ch1.pos[batch].assign( tf.gather(self.ch1.pos[batch],idx,axis=1) )
            self.ch2.pos[batch].assign( tf.gather(self.ch2.pos[batch],idx,axis=1) )
            self.ch2_original.pos[batch].assign( tf.gather(self.ch2_original.pos[batch],idx,axis=1) )
            self.ch1.frame[batch].assign( tf.gather(self.ch1.frame[batch],idx,axis=1) )
            self.ch2.frame[batch].assign( tf.gather(self.ch2.frame[batch],idx,axis=1) )
            self.ch2_original.frame[batch].assign( tf.gather(self.ch2_original.frame[batch],idx,axis=1) )
        
        
    #%% Split dataset or load subset
    def SubsetWindow(self, subset):
    # loading subset of dataset by creating a window of size subset 
        print('Taking a subset of size',subset,'...')
        if len(self.ch1.pos)>1: raise ValueError('SubsetRandom should be used before creating batches!')
        
        self.img, self.imgsize, self.mid = self.imgparams()
        l_grid = self.mid - np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2
        r_grid = self.mid + np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2
    
        mask1 = np.where( (self.ch1.pos[0][:,0] >= l_grid[0]) * (self.ch1.pos[0][:,1] >= l_grid[1])
                            * (self.ch1.pos[0][:,0] <= r_grid[0]) * (self.ch1.pos[0][:,1] <= r_grid[1]) , True, False)
        mask2 = np.where( (self.ch2.pos[0][:,0] >= l_grid[0]) * (self.ch2.pos[0][:,1] >= l_grid[1])
                            * (self.ch2.pos[0][:,0] <= r_grid[0]) * (self.ch2.pos[0][:,1] <= r_grid[1]), True, False )

        self = self.gather(mask1, mask2)
        
        
    def SubsetRandom(self, subset):
    # loading subset of dataset by taking a random subset
        if len(self.ch1.pos)>1: raise ValueError('SubsetRandom should be used before creating batches!')
        if self.linked:
            mask1=np.random.choice(self.ch1.pos[0].shape[0], int(self.ch1.pos[0].shape[0]*subset))
            mask2=mask1
        else:
            mask1=np.random.choice(self.ch1.pos[0].shape[0], int(self.ch1.pos[0].shape[0]*subset))
            mask2=np.random.choice(self.ch2.pos[0].shape[0], int(self.ch2.pos[0].shape[0]*subset))
            
        self = self.gather(mask1, mask2)
        
        
    def SplitDataset(self):
    # Splits dataset into 2 halves for cross validation
        if len(self.ch1.pos)>1: raise ValueError('SplitDataset should be used before creating batches!')
        if self.Neighbours: print('WARNING: splitting datasets means the neighbours need to be reloaded!')
        
        N1=self.ch1.pos[0].shape[0]
        N2=self.ch2.pos[0].shape[0]
        if self.linked:
            if N1!=N2: raise Exception('Datasets are linked but not equal in size')
            mask1=np.ones(N1, dtype=bool)
            mask1[int(N1/2):]=False
            np.random.shuffle(mask1)  # create random mask to split dataset in two
            mask2 = np.abs(mask1-1).astype('bool')

            other1=self.gather(mask1, mask1)
            other2=self.gather(mask2, mask2)
            
        else:
            mask11=np.ones(N1, dtype=bool)
            mask11[int(N1/2):]=False
            mask12=np.ones(N2, dtype=bool)
            mask12[int(N2/2):]=False
            np.random.shuffle(mask11)  # create random mask to split dataset in two
            np.random.shuffle(mask12)
            mask21 = np.abs(mask11-1).astype('bool')
            mask22 = np.abs(mask12-1).astype('bool')
            
            other1=self.gather(mask11, mask12)
            other2=self.gather(mask21, mask22)
        
        return other1, other2
    
    
    def SplitFrames(self):
    # splits frames of both ch1 and ch2
        self.ch1.SplitFrames()
        self.ch2.SplitFrames()
        self.ch2_original.SplitFrames()
        
    
    def gather(self, idx1, idx2):
    # gathers the indexes of both Channels
        other = copy.deepcopy(self)
        
        del other.ch1, other.ch2, other.ch2_original
        other.ch1 = Channel(pos=tf.gather(self.ch1.pos[0],idx1,axis=0), frame=self.ch1.frame[0][idx1])
        other.ch2 = Channel(pos=tf.gather(self.ch2.pos[0],idx2,axis=0), frame=self.ch2.frame[0][idx2])
        other.ch2_original = Channel(pos=tf.gather(self.ch2_original.pos[0],idx2,axis=0), frame=self.ch2.frame[0][idx2])
        return other
    
    
    #%% Generate Neighbours
    def find_neighbours(self, maxDistance=50, k=20):
    # Tries to generate neighbours according to brightest spots, and tries kNN otherwise
        print('Finding neighbours within a distance of',maxDistance,'nm for spots containing at least',k,'neighbours...')
        (idx1list, idx2list) = ([],[])
        for batch in range(len(self.ch1.pos)):
            try:
                idx1, idx2 = self.find_BrightNN(self.ch1.pos[batch].numpy(), self.ch2.pos[batch].numpy(), maxDistance=maxDistance, threshold=k)
            except Exception:
                print('Not enough bright Neighbours found in current setting. Switching to kNN with k = ',k,'!')
                idx1, idx2 = self.find_kNN(self.ch1.pos[batch].numpy(), self.ch2.pos[batch].numpy(), k)
            idx1list.append(idx1)
            idx2list.append(idx2)
            
        self.ch1.load_NN_matrix(idx1list)
        self.ch2.load_NN_matrix(idx2list)
        self.Neighbours=True
        
        
    def find_BrightNN(self, pos1, pos2, maxDistance = 50, threshold = 20):
    # generates the brightest neighbours
    # outputs a list of indices for the neigbhours
        with Context() as ctx: # loading all NN
            counts,indices = PostProcessMethods(ctx).FindNeighbors(pos1, pos2, maxDistance)
    
        ## putting all NNidx in a list 
        (idxlist, pos, i) = ([], 0,0)
        for count in counts:
            idxlist.append( np.stack([
                i * np.ones([count], dtype=int),
                indices[pos:pos+count] 
                ]) )
            pos+=count
            i+=1
            
        ## filtering the NNidx list to be square
        (idx1list, idx2list) = ([], [])
        for idx in idxlist:
            if idx.size>0 and idx.shape[1] > threshold: # filter for brightest spots above threshold
                # select random sample from brightest spost
                idx1list.append(idx[0, np.random.choice(idx.shape[1], threshold)]) 
                idx2list.append(idx[1, np.random.choice(idx.shape[1], threshold)]) 
    
        ## look if neighbours actually have been found
        if idx1list==[]: 
            raise Exception('No neighbours were generated. Adjust the threshold or maxDistance!')
        else:
            self.NN_maxDist=maxDistance
            self.NN_threshold=threshold
            
        ## Loading the indexes as matrices
        return idx1list, idx2list
        
        
    def find_kNN(self, pos1, pos2, k):
    # generates the k-nearest neighbours
    # outputs a list of indices for the neigbhours
        (idx1list, idx2list) = ([],[])
        for i in range(pos1.shape[0]):
            idx1list.append( (i * np.ones(k, dtype=int)) )
        for loc in pos1:
            distances = np.sum((loc - pos2)**2 , axis = 1)
            idx2list.append( np.argsort( distances )[:k] )
            
        self.NN_k=k
        
        ## Loading the indexes as matrices
        return idx1list, idx2list
        
    