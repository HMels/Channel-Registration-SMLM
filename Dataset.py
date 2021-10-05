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
    def __init__(self, path, pix_size=1, linked=False, imgshape=[512, 512]):
        self.path=path
        self.pix_size=pix_size
        self.imgshape=imgshape
        self.linked=linked        
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
        
        self.ch2_original=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams()                     # loading the image parameters
        self.center_image()
    
    
    def load_dataset_hdf5(self):
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
        
        self.ch1 = Channel(pos = ch1.pos* self.pix_size, frame = ch1.frame)
        self.ch2 = Channel(pos = ch2.pos* self.pix_size, frame = ch2.frame)
        
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
        #for batch in range(len(self.ch1.pos)):
        self.ch1.pos.assign(self.ch1.pos - self.mid[None,:])
        self.ch2.pos.assign(self.ch2.pos - self.mid[None,:])
        self.ch2_original.pos.assign(self.ch2_original.pos - self.mid[None,:])
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.mid = tf.Variable([0,0], dtype=tf.float32)
        
        
    #%% pair_functions
    #@tf.function
    def link_dataset(self, FrameLinking=False):
    # links dataset with a simple iterative nearest neighbour method
    # FrameLinking links the dataset per frame
        print('Linking datasets...')
        #if len(self.ch1.pos)>1: raise Exception('Cannot link after splitting!')
        ch1_frame=self.ch1.frame.numpy()
        ch2_frame=self.ch2.frame.numpy()
        ch1_pos=self.ch1.pos.numpy()
        ch2_pos=self.ch2.pos.numpy()
        
        (locsB,frameB)=([],[])
        for i in range(ch1_pos.shape[0]):
            if FrameLinking:
                sameframe_pos = ch2_pos[ch2_frame==ch1_frame[i],:]
                dists = np.sqrt(np.sum((ch1_pos[i,:]-sameframe_pos)**2,1))
            
                iB=np.argmin(dists)
                locsB.append(sameframe_pos[iB,:])
                frameB.append(ch1_frame[i])
            else:
                dists = np.sqrt(np.sum((ch1_pos[i,:]-ch2_pos)**2,1))
                iB=np.argmin(dists)
                locsB.append(ch2_pos[iB,:])
                frameB.append(ch2_frame[iB])
            
        if not locsB: raise ValueError('When Coupling Datasets, one of the Channels returns empty')
        
        del self.ch2
        self.ch2 = Channel(locsB, frameB)
        self.linked = True
        
        
    def Filter_Pairs(self, maxDist=150):
    # Filter pairs above maxDist
        print('Filtering pairs above',maxDist,'nm...')
        if not self.linked: raise Exception('Dataset should be linked before filtering pairs')
        
        dists = np.sqrt(np.sum( (self.ch1.pos.numpy() - self.ch2.pos.numpy())**2 ,axis=1))
        idx = np.argwhere(dists<maxDist)
        if idx.shape[0]==0: raise ValueError('All localizations will be filtered out in current settings.')
        
        ch1_pos = self.ch1.pos.numpy()[idx[:,0],:]
        ch2_pos = self.ch2.pos.numpy()[idx[:,0],:]
        ch2_original_pos = self.ch2_original.pos.numpy()[idx[:,0],:]
        ch1_frame = self.ch1.frame.numpy()[idx[:,0]]
        ch2_frame = self.ch2.frame.numpy()[idx[:,0]]
        ch2_original_frame = self.ch2_original.frame.numpy()[idx[:,0]]
        
        del self.ch1, self.ch2, self.ch2_original
        self.ch1 = Channel(ch1_pos, ch1_frame)
        self.ch2 = Channel(ch2_pos, ch2_frame)
        self.ch2_original = Channel(ch2_original_pos, ch2_original_frame)
        
        
        
    #%% Split dataset or load subset
    def SubsetWindow(self, subset):
    # loading subset of dataset by creating a window of size subset 
        print('Taking a subset of size',subset,'...')
        #if len(self.ch1.pos)>1: raise ValueError('SubsetRandom should be used before creating batches!')
        
        self.img, self.imgsize, self.mid = self.imgparams()
        l_grid = self.mid - np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2
        r_grid = self.mid + np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2
    
        mask1 = np.where( (self.ch1.pos[:,0] >= l_grid[0]) * (self.ch1.pos[:,1] >= l_grid[1])
                            * (self.ch1.pos[:,0] <= r_grid[0]) * (self.ch1.pos[:,1] <= r_grid[1]) , True, False)
        mask2 = np.where( (self.ch2.pos[:,0] >= l_grid[0]) * (self.ch2.pos[:,1] >= l_grid[1])
                            * (self.ch2.pos[:,0] <= r_grid[0]) * (self.ch2.pos[:,1] <= r_grid[1]), True, False )

        self = self.gather(mask1, mask2)
        
        
    def SubsetRandom(self, subset):
    # loading subset of dataset by taking a random subset
        #if len(self.ch1.pos)>1: raise ValueError('SubsetRandom should be used before creating batches!')
        if self.linked:
            mask1=np.random.choice(self.ch1.pos.shape[0], int(self.ch1.pos.shape[0]*subset))
            mask2=mask1
        else:
            mask1=np.random.choice(self.ch1.pos.shape[0], int(self.ch1.pos.shape[0]*subset))
            mask2=np.random.choice(self.ch2.pos.shape[0], int(self.ch2.pos.shape[0]*subset))
            
        self = self.gather(mask1, mask2)
        
        
    def SplitDataset(self):
    # Splits dataset into 2 halves for cross validation
        #if len(self.ch1.pos)>1: raise ValueError('SplitDataset should be used before creating batches!')
        if self.Neighbours: print('WARNING: splitting datasets means the neighbours need to be reloaded!')
        
        N1=self.ch1.pos.shape[0]
        N2=self.ch2.pos.shape[0]
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
        other.ch1 = Channel(pos=tf.gather(self.ch1.pos,idx1,axis=0), frame=self.ch1.frame[idx1])
        other.ch2 = Channel(pos=tf.gather(self.ch2.pos,idx2,axis=0), frame=self.ch2.frame[idx2])
        other.ch2_original = Channel(pos=tf.gather(self.ch2_original.pos,idx2,axis=0), frame=self.ch2.frame[idx2])
        return other
    
    
    #%% Generate Neighbours
    def find_neighbours(self, maxDistance=50, k=20, FrameLinking=False):
    # Tries to generate neighbours according to brightest spots, and tries kNN otherwise
        print('Finding neighbours within a distance of',maxDistance,'nm for spots containing at least',k,'neighbours...')
        ch1_frame=self.ch1.frame
        ch2_frame=self.ch2.frame
        ch1_pos=self.ch1.pos
        ch2_pos=self.ch2.pos
        
        
        if FrameLinking: ## Neighbours per frame
            frame,_=tf.unique(self.ch1.frame)
            (pos1, frame1, pos2, frame2) = ([],[],[],[])
            for fr in frame:
                framepos1 = ch1_pos.numpy()[ch1_frame.numpy()==fr,:]
                framepos2 = ch2_pos.numpy()[ch2_frame.numpy()==fr,:]
                
                try: ################# idx1_fr and idx2_fr do not refer to ch1 and ch2
                    idx1, idx2 = self.find_BrightNN(framepos1, framepos2, maxDistance=maxDistance, threshold=k)
                except Exception:
                    idx1, idx2 = self.find_kNN(framepos1, framepos2, k)
                    
                p1, f1 = self.load_NN_matrix(idx1, ch1_pos, ch1_frame)
                p2, f2 = self.load_NN_matrix(idx2, ch2_pos, ch2_frame)
                pos1.append(p1)
                pos2.append(p2)
                frame1.append(f1)
                frame2.append(f2)
            
            pos1=tf.concat(pos1, axis=0)
            pos2=tf.concat(pos2, axis=0)
            frame1=tf.concat(frame1, axis=0)
            frame2=tf.concat(frame2, axis=0)
            
        else: ## taking the whole dataset as a single batch   
            try:
                idx1, idx2 = self.find_BrightNN(ch1_pos.numpy(), ch2_pos.numpy(), maxDistance=maxDistance, threshold=k)
            except Exception:
                #print('Not enough bright Neighbours found in current setting. Switching to kNN with k = ',k,'!')
                idx1, idx2 = self.find_kNN(ch1_pos.numpy(), ch2_pos.numpy(), k)
            
            pos1, frame1 = self.load_NN_matrix(idx1, ch1_pos, ch1_frame)
            pos2, frame2 = self.load_NN_matrix(idx2, ch2_pos, ch2_frame)
        
        self.ch1NN = Channel( pos1, frame1 )
        self.ch2NN = Channel( pos2, frame2 )
        self.Neighbours=True
        
        
    def load_NN_matrix(self, idx, pos, frame):
        (NN,fr)=([],[])
        for nn in idx:
            NN.append(tf.gather(pos,nn,axis=0))
            fr.append(tf.gather(frame,nn[0]))
        return tf.stack(NN, axis=0) , tf.stack(fr, axis=0)
        
        
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
        return idx1list, idx2list
        
    