# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:17:52 2021

@author: Mels
"""
import copy
import pandas as pd
import tensorflow as tf
import numpy as np
from photonpy import PostProcessMethods, Context, Dataset

from Channel import Channel        
from Registration import Registration


#%% Dataset
class dataset(Registration):
    def __init__(self, path, pix_size=1, loc_error=10, linked=False, imgshape=[512, 512], 
                 FrameLinking=True, FrameOptimization=False):
        self.path=path            # the string or list containing the strings of the file location of the dataset
        self.pix_size=pix_size    # the multiplicationfactor to change the dataset into units of nm
        self.loc_error=loc_error  # localization error
        self.imgshape=imgshape    # number of pixels of the dataset
        self.linked=linked        # is the data linked/paired?
        self.FrameLinking=FrameLinking              # will the dataset be linked or NN per frame?
        self.FrameOptimization=FrameOptimization    # will the dataset be optimized per frame
        self.subset=1       
        Registration.__init__(self)
        
        
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
    
        self.ch1 = Channel(pos = data1[:,:2]* self.pix_size, frame = data1[:,2])
        self.ch2 = Channel(pos = data2[:,:2]* self.pix_size, frame = data2[:,2])
        
        self.ch20=copy.deepcopy(self.ch2)          # original channel 2
        self.img, self.imgsize, self.mid = self.imgparams()      # loading the image parameters
        self.center_image()
    
    
    def load_dataset_hdf5(self, align_rcc=True):
        ## Loading dataset
        if len(self.path)==1 or isinstance(self.path,str):
            # Dataset is grouped, meaning it has to be split manually
            print('Loading dataset... \n Grouping...')
            ds = Dataset.load(self.path[0],saveGroups=True)
            ch1 = ds[ds.group==0]
            ch2 = ds[ds.group==1]
        elif len(self.path)==2:
            # Dataset consists over 2 files
            print('Loading dataset...')
            ch1 = Dataset.load(self.path[0])
            ch2 = Dataset.load(self.path[1])
        else:
            raise TypeError('Path invalid')
            
        if align_rcc:
            print('Alignning both datasets')
            shift = Dataset.align(ch1, ch2)
            print('RCC shift equals', shift*self.pix_size)
            if not np.isnan(shift).any():
                ch1.pos+= shift
            else: 
                print('Warning: Shift contains infinities')
        
        self.ch1 = Channel(pos = ch1.pos* self.pix_size, frame = ch1.frame)
        self.ch2 = Channel(pos = ch2.pos* self.pix_size, frame = ch2.frame)
        
        self.ch20=copy.deepcopy(self.ch2)
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
                           [ tf.reduce_max([img1[1,0], img2[1,0]]),  tf.reduce_max([img1[1,1], img2[1,1]]) ]], 
                          dtype=tf.float32)

        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
    
    
    def center_image(self):
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.ch1.pos.assign(self.ch1.pos - self.mid[None,:])
        self.ch2.pos.assign(self.ch2.pos - self.mid[None,:])
        self.ch20.pos.assign(self.ch20.pos - self.mid[None,:])
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.mid = tf.Variable([0,0], dtype=tf.float32)
        
        
    def center_channels(self):
        self.ch1.center()
        self.ch2.center()
        self.ch20.center()
        
        
    #%% Split dataset or load subset
    def SubsetWindow(self, subset, linked=None):
    # loading subset of dataset by creating a window of size subset 
        if linked is None: linked=self.linked
        print('Taking a subset of size',subset,'...')
        
        self.img, self.imgsize, self.mid = self.imgparams()
        l_grid = self.mid - np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2
        r_grid = self.mid + np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2

        idx1 = (np.where(self.ch1.pos.numpy()[:,0] >= l_grid[0],True,False) * np.where(self.ch1.pos.numpy()[:,1] >= l_grid[1],True,False)
                            * np.where(self.ch1.pos.numpy()[:,0] <= r_grid[0],True,False) * np.where(self.ch1.pos.numpy()[:,1] <= r_grid[1],True,False) )
        idx2 = (np.where(self.ch2.pos.numpy()[:,0] >= l_grid[0],True,False) * np.where(self.ch2.pos.numpy()[:,1] >= l_grid[1],True,False)
                            * np.where(self.ch2.pos.numpy()[:,0] <= r_grid[0],True,False) * np.where(self.ch2.pos.numpy()[:,1] <= r_grid[1],True,False) )
        
        self.subset*=subset
        if linked:
            idx=idx1*idx2
            return self.gather( np.argwhere(idx), np.argwhere(idx))
        else:
            return self.gather( np.argwhere(idx1), np.argwhere(idx2))
        
        
    def SubsetRandom(self, subset):
    # loading subset of dataset by taking a random subset
        if self.linked:
            mask1=self.random_choice(self.ch1.pos.shape[0], int(self.ch1.pos.shape[0]*subset))
            mask2=mask1
        else:
            mask1=self.random_choice(self.ch1.pos.shape[0], int(self.ch1.pos.shape[0]*subset))
            mask2=self.random_choice(self.ch2.pos.shape[0], int(self.ch2.pos.shape[0]*subset))
        self.subset*=subset
            
        return self.gather(np.argwhere(mask1), np.argwhere(mask2))
        
        
    def SplitDataset(self, linked=None):
    # Splits dataset into 2 halves for cross validation)
        if linked is None: linked=self.linked
        if self.Neighbours: print('WARNING: splitting datasets means the neighbours need to be reloaded!')
        
        N1=self.ch1.pos.shape[0]
        N2=self.ch2.pos.shape[0]
        if linked:
            if N1!=N2: raise Exception('Datasets are linked but not equal in size')
            mask1=np.ones(N1, dtype=bool)
            mask1[int(N1/2):]=False
            np.random.shuffle(mask1)  # create random mask to split dataset in two
            
            idx1=np.argwhere(mask1)
            idx2=np.argwhere( (mask1-1).astype(bool))
            other1=self.gather(idx1, idx1)
            other2=self.gather(idx2, idx2)
            
        else:
            mask1=np.ones(N1, dtype=bool)
            mask1[int(N1/2):]=False
            mask2=np.ones(N2, dtype=bool)
            mask2[int(N2/2):]=False
            np.random.shuffle(mask1)  # create random mask to split dataset in two
            np.random.shuffle(mask2)
            
            other1=self.gather(np.argwhere(mask1), np.argwhere(mask2) )
            other2=self.gather(np.argwhere((mask1-1).astype('bool')), np.argwhere((mask2-1).astype('bool')))
        
        return other1, other2
        
    
    def gather(self, idx1, idx2):
    # gathers the indexes of both Channels
        other = copy.deepcopy(self)
        del other.ch1, other.ch2, other.ch20
        other.ch1 = Channel(pos=tf.gather_nd(self.ch1.pos,idx1), frame=tf.gather_nd(self.ch1.frame,idx1))
        other.ch2 = Channel(pos=tf.gather_nd(self.ch2.pos,idx2), frame=tf.gather_nd(self.ch2.frame,idx2))
        other.ch20 = Channel(pos=tf.gather_nd(self.ch20.pos,idx2), frame=tf.gather_nd(self.ch2.frame,idx2))
        return other
    
    
    #%% pair_functions
    def link_dataset(self, maxDist=1000,FrameLinking=None):
        print('Linking Datasets for localizations within a distance of',maxDist,'nm...')
        if FrameLinking is None: FrameLinking=self.FrameLinking
        ch1_frame=self.ch1.frame.numpy()
        ch2_frame=self.ch2.frame.numpy()
        ch20_frame=self.ch20.frame.numpy()
        ch1_pos=self.ch1.pos.numpy()
        ch2_pos=self.ch2.pos.numpy()
        ch20_pos=self.ch20.pos.numpy()
        
        (pos1, frame1, pos2, frame2, pos20, frame20) = ([],[],[],[],[],[])
        if FrameLinking: ## Linking per frame
            frame,_=tf.unique(self.ch1.frame)
            for fr in frame:
                # Generate neighbouring indices per frame
                framepos1 = ch1_pos[ch1_frame==fr,:]
                framepos2 = ch2_pos[ch2_frame==fr,:]
                framepos20 = ch20_pos[ch20_frame==fr,:]
                idxlist = self.FindNeighbours_idx(framepos1, framepos2, maxDist=maxDist)
                
                for idx in idxlist:
                    if len(idx[0])!=0:
                        posA=framepos1[idx[0][0],:]
                        posB=framepos2[idx[1],:]
                        posB0=framepos20[idx[1],:]
                        frame1.append(fr)
                        frame2.append(fr)
                        frame20.append(fr)
                        pos1.append(posA)
                        pos2.append(posB[ np.argmin( np.sum((posA-posB)**2) ) ,:])
                        pos20.append(posB0[ np.argmin( np.sum((posA-posB)**2) ) ,:])
            
        else: ## taking the whole dataset as a single batch
            idxlist = self.FindNeighbours_idx(ch1_pos, ch2_pos, maxDist=maxDist)
            for idx in idxlist:
                if len(idx[0])!=0:
                    i=idx[0][0]
                    posB=ch2_pos[idx[1],:]
                    posB0=ch20_pos[idx[1],:]
                    j=np.argmin( np.sum((ch1_pos[i,:]-posB)**2) )
                    
                    frame1.append(ch1_frame[i])
                    frame2.append(ch2_frame[idx[1][j]])
                    frame20.append(ch20_frame[idx[1][j]])
                    pos1.append(ch1_pos[i,:])
                    pos2.append(posB[j,:])
                    pos20.append(posB0[j,:])
        
        if len(pos1)==0 or len(pos2)==0: raise ValueError('When Coupling Datasets, one or both of the Channels returns empty')
        
        del self.ch1, self.ch2, self.ch20
        self.ch1 = Channel( np.array(pos1) , np.array(frame1) )
        self.ch2 = Channel( np.array(pos2) , np.array(frame2) )
        self.ch20 = Channel( np.array(pos20) , np.array(frame20) )
        self.linked = True
        
        
    #%% Filter
    def Filter(self, maxDist):
    # The function for filtering both pairs and neigbhours 
        if self.linked: self.Filter_Pairs(maxDist)
        if self.Neighbours: self.Filter_Neighbours(maxDist)
        
        
    def Filter_Pairs(self, maxDist=150):
    # Filter pairs above maxDist
        print('Filtering pairs above',maxDist,'nm...')
        if not self.linked: raise Exception('Dataset should be linked before filtering pairs!')
        N0=self.ch1.pos.shape[0]
        
        dists = np.sqrt(np.sum( (self.ch1.pos.numpy() - self.ch2.pos.numpy())**2 , axis=1))
        idx = np.argwhere(dists<maxDist)
        
        ch1_pos = self.ch1.pos.numpy()[idx[:,0],:]
        ch2_pos = self.ch2.pos.numpy()[idx[:,0],:]
        ch20_pos = self.ch20.pos.numpy()[idx[:,0],:]
        ch1_frame = self.ch1.frame.numpy()[idx[:,0]]
        ch2_frame = self.ch2.frame.numpy()[idx[:,0]]
        ch20_frame = self.ch20.frame.numpy()[idx[:,0]]
        
        if ch1_pos.shape[0]==0: raise Exception('All positions will be filtered out in current settings!')
        del self.ch1, self.ch2, self.ch20
        self.ch1 = Channel(ch1_pos, ch1_frame)
        self.ch2 = Channel(ch2_pos, ch2_frame)
        self.ch20 = Channel(ch20_pos, ch20_frame)
        N1=self.ch1.pos.shape[0]
        print('Out of the '+str(N0)+' pairs localizations, '+str(N0-N1)+' have been filtered out ('+str(round((1-(N1/N0))*100,1))+'%)')
        
        
    def Filter_Neighbours(self, maxDist=150):
        print('Filtering localizations that have no Neighbours under',maxDist,'nm...')
        if not self.Neighbours: raise Exception('Tried to filter without the Neighbours having been generated')
        N0=self.ch1NN.pos.shape[0]
        
        dists = np.sqrt(np.sum( (self.ch1NN.pos.numpy() - self.ch2NN.pos.numpy())**2 , axis=1))
        idx = np.argwhere(dists<maxDist)
        ch1_pos = self.ch1NN.pos.numpy()[idx[:,0],:]
        ch2_pos = self.ch2NN.pos.numpy()[idx[:,0],:]
        ch1_frame = self.ch1NN.frame.numpy()[idx[:,0]]
        ch2_frame = self.ch2NN.frame.numpy()[idx[:,0]]
        
        if ch1_pos.shape[0]==0: raise Exception('All positions will be filtered out in current settings!')
        del self.ch1NN, self.ch2NN
        self.ch1NN = Channel(ch1_pos, ch1_frame)
        self.ch2NN = Channel(ch2_pos, ch2_frame)
        N1=self.ch1NN.pos.shape[0]
        print('Out of the '+str(N0)+' Neighbours localizations, '+str(N0-N1)+' have been filtered out ('+str(round((1-(N1/N0))*100,1))+'%)')
        
        
    #%% Generate Neighbours
    def find_neighbours(self, maxDistance=50, FrameLinking=None):
    # Tries to generate neighbours according to all spots
        print('Finding neighbours within a distance of',maxDistance,'nm for spots.')
        if FrameLinking is None: FrameLinking=self.FrameLinking
        maxDistance=np.float32(maxDistance)
        self.NN_maxDist=maxDistance
        
        with Context() as ctx: # loading all NN
            counts,indices = PostProcessMethods(ctx).FindNeighbors(self.ch1.pos.numpy(), 
                                                                   self.ch2.pos.numpy(), maxDistance)
    
        ## putting all NNidx in a list 
        (pos1, pos2, frame1, pos, i) = ([], [], [], 0, 0)
        for count in counts:
            pos1.append(tf.gather(self.ch1.pos, i * np.ones([count], dtype=int)))
            pos2.append(tf.gather(self.ch2.pos, indices[pos:pos+count]))
            frame1.append(tf.gather(self.ch1.frame, i * np.ones([count], dtype=int)))
            pos+=count
            i+=1
        
        # load matrix as Channel class
        pos1=tf.concat(pos1, axis=0)
        pos2=tf.concat(pos2, axis=0)
        frame1=tf.concat(frame1, axis=0)
        self.ch1NN = Channel( pos1, frame1 )
        self.ch2NN = Channel( pos2, frame1 )
        self.Neighbours=True
            
    
    def FindNeighbours_idx(self, pos1, pos2, maxDist):
    # prints a list containing the neighbouring indices of two channels
        with Context() as ctx: # loading all NN
            counts,indices = PostProcessMethods(ctx).FindNeighbors(pos1, pos2, maxDist)
    
        ## putting all NNidx in a list 
        (idxlist, pos, i) = ([], 0,0)
        for count in counts:
            idxlist.append( np.stack([
                i * np.ones([count], dtype=int),
                indices[pos:pos+count] 
                ]) )
            pos+=count
            i+=1
        return idxlist
    
        
    def random_choice(self,original_length, final_length):
        lst=[]
        while len(lst)<final_length:
            r=np.random.randint(0,original_length)
            if r not in lst: lst.append(r)
        return lst