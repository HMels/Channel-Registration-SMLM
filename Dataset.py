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

from Registration import Registration
from Channel import Channel 


#%% Dataset
class dataset(Registration):
    def __init__(self, path, pix_size=1, loc_error=10, mu=0, coloc_error=None, imgshape=[512, 512],
                 linked=False, FrameLinking=True, BatchOptimization=False, execute_linked=True):
        self.path=path            # the string or list containing the strings of the file location of the dataset
        self.pix_size=pix_size    # the multiplicationfactor to change the dataset into units of nm
        self.loc_error=loc_error  # localization error
        self.coloc_error=coloc_error if coloc_error is not None else np.sqrt(2)*loc_error if loc_error is not None else None
        self.mu=mu
        self.imgshape=imgshape    # number of pixels of the dataset
        self.linked=linked        # is the data linked/paired?
        self.linked_original=linked
        self.FrameLinking=FrameLinking              # will the dataset be linked or NN per frame?
        self.BatchOptimization=BatchOptimization    # will the dataset be optimized per frame
        self.execute_linked=execute_linked          # false if optimization via NN
        self.counts_linked=None
        self.counts_Neighbours=None
        Registration.__init__(self)
        
        
    def reload_dataset(self):
        # reloads the original channels of the dataset
        self.linked=self.linked_original
        del self.ch1, self.ch2, self.ch20linked
        self.ch1=copy.deepcopy(self.ch10)
        self.ch2=copy.deepcopy(self.ch20) 
        self.ch20linked=copy.deepcopy(self.ch20) 
        try: 
            del self.ch1NN, self.ch2NN
        except: pass
        self.Neighbours=False
        
        
    def ClusterDataset(self, loc_error=None):
        # outputs a dataset of cluster center of masses
        other=copy.deepcopy(self)
        pos1=self.ch1.ClusterCOM()[0]
        pos2=self.ch2.ClusterCOM()[0]
        del other.ch1, other.ch2, other.ch20linked, other.ch10, other.ch20
        other.ch1=Channel(pos1, np.ones(pos1.shape[0]))
        other.ch2=Channel(pos2, np.ones(pos2.shape[0]))
        other.ch10=Channel(pos1, np.ones(pos1.shape[0]))
        other.ch20=Channel(pos2, np.ones(pos2.shape[0]))
        other.ch20linked=Channel(pos2, np.ones(pos2.shape[0]))
        other.loc_error=loc_error if loc_error is not None else None
        other.coloc_error=np.sqrt(2)*loc_error if loc_error is not None else None
        return other
        
        
    #%% load_dataset
    def load_dataset_excel(self, transpose=False):
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
        self.ch10=copy.deepcopy(self.ch1)
        self.ch20=copy.deepcopy(self.ch2)
        self.ch20linked=copy.deepcopy(self.ch2)
        if transpose:
            self.ch1.transpose_axis()
            self.ch2.transpose_axis()
            self.ch10.transpose_axis()
            self.ch20.transpose_axis()
            self.ch20linked.transpose_axis()
        
        self.img, self.imgsize, self.mid = self.imgparams()      # loading the image parameters
        self.center_image()
    
    
    def load_dataset_hdf5(self, align_rcc=True, transpose=False):
        ## Loading dataset
        if len(self.path)==1 or isinstance(self.path,str):
            # Dataset is grouped, meaning it has to be split manually
            print('Loading dataset...')
            print(self.path[0])
            ds = Dataset.load(self.path[0],saveGroups=True)
            print('\n Grouping...')
            ch1 = ds[ds.group==0]
            ch2 = ds[ds.group==1]
        elif len(self.path)==2:
            # Dataset consists over 2 files
            print('Loading dataset...')
            print(self.path[0])
            print(self.path[1])
            ch1 = Dataset.load(self.path[0])
            ch2 = Dataset.load(self.path[1])
        else:
            raise TypeError('Path invalid')
            
        if transpose:
            ch1.pos=ch1.pos[:,[1,0]]
            ch2.pos=ch2.pos[:,[1,0]]
        
        if align_rcc:
            print('Alignning both datasets')
            shift = Dataset.align(ch1, ch2)
            print('RCC shift equals', shift*self.pix_size)
            if not np.isnan(shift).any():
                ch1.pos+= shift
            else: 
                print('Warning: Shift contains infinities')
        
        self.ch1 = Channel(pos = ch1.pos* self.pix_size, frame = ch1.frame, group=ch1.group)
        self.ch2 = Channel(pos = ch2.pos* self.pix_size, frame = ch2.frame, group=ch2.group)
        self.ch10=copy.deepcopy(self.ch1)
        self.ch20=copy.deepcopy(self.ch2)
        self.ch20linked=copy.deepcopy(self.ch2)
        
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
        self.ch10.pos.assign(self.ch10.pos - self.mid[None,:])
        self.ch20.pos.assign(self.ch20.pos - self.mid[None,:])
        self.ch20linked.pos.assign(self.ch20linked.pos - self.mid[None,:])
        self.img, self.imgsize, self.mid = self.imgparams() 
        self.mid = tf.Variable([0,0], dtype=tf.float32)
        
        
    def center_channels(self):
        self.ch1.center()
        self.ch2.center()
        self.ch10.center()
        self.ch20.center()
        self.ch20linked.center()
        
        
    #%% Split dataset or load subset
    def AppendDataset(self, other):
        self.ch1.AppendChannel(other.ch1)
        if self.ch10 is not None and other.ch10 is not None: self.ch10.AppendChannel(other.ch10)
        if self.ch2 is not None and other.ch2 is not None: self.ch2.AppendChannel(other.ch2)
        if self.ch20 is not None and other.ch20 is not None: self.ch20.AppendChannel(other.ch20)
        if self.ch20linked is not None and other.ch20linked is not None:
            self.ch20linked.AppendChannel(other.ch20linked)
        
    
    def SubsetWindow(self, window=None, subset=None, linked=None):
    # loading subset of dataset by creating a window of size subset 
        if linked is None: linked=self.linked        
        self.img, self.imgsize, self.mid = self.imgparams()
        if subset is not None:
            window = np.array([self.mid - np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2,
                        self.mid + np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2       
                               ])
        elif window is None:
            raise Exception('No window or subset selected')
            
        idx1=(np.where(self.ch1.pos.numpy()[:,0]>window[0,0],True,False)
             *np.where(self.ch1.pos.numpy()[:,0]<window[0,1],True,False)
             *np.where(self.ch1.pos.numpy()[:,1]>window[1,0],True,False)
             *np.where(self.ch1.pos.numpy()[:,1]<window[1,1],True,False))
        idx2=(np.where(self.ch2.pos.numpy()[:,0]>window[0,0],True,False)
             *np.where(self.ch2.pos.numpy()[:,0]<window[0,1],True,False)
             *np.where(self.ch2.pos.numpy()[:,1]>window[1,0],True,False)
             *np.where(self.ch2.pos.numpy()[:,1]<window[1,1],True,False))
        
        if self.gridsize is not None:
            self.x1_min=window[0,0]/self.gridsize
            self.x1_max=window[0,1]/self.gridsize
            self.x2_min=window[1,0]/self.gridsize
            self.x2_max=window[1,1]/self.gridsize
            
        if len(idx1)==0 or len(idx2)==0: raise  ValueError('No values returned after subset')
        if linked:
            idx=idx1*idx2
            return self.gather(np.argwhere(idx), np.argwhere(idx))
        else:
            return self.gather(np.argwhere(idx1), np.argwhere(idx2))
        
        
    def SubsetRandom(self, subset):
    # loading subset of dataset by taking a random subset
        if self.linked:
            mask1=self.random_choice(self.ch1.pos.shape[0], int(self.ch1.pos.shape[0]*subset))
            mask2=mask1
        else:
            mask1=self.random_choice(self.ch1.pos.shape[0], int(self.ch1.pos.shape[0]*subset))
            mask2=self.random_choice(self.ch2.pos.shape[0], int(self.ch2.pos.shape[0]*subset))
            
        if len(mask1)==0 or len(mask2)==0: raise  ValueError('No values returned after subset')
        return self.gather(np.argwhere(mask1), np.argwhere(mask2))
    
    
        
    
    def SubsetFrames(self, begin_frames, end_frames):
    # select a certain subset of frames
        if end_frames<begin_frames: raise ValueError('Invalid input')
        idx1=(np.where(self.ch1.frame<end_frames,True,False)*
                         np.where(self.ch1.frame>=begin_frames,True,False))
        idx2=(np.where(self.ch2.frame<end_frames,True,False)*
                         np.where(self.ch2.frame>=begin_frames,True,False))
        
        if len(idx1)==0 or len(idx2)==0: raise  ValueError('No values returned after subset')
        if self.linked:
            idx=idx1*idx2
            return self.gather(np.argwhere(idx), np.argwhere(idx))
        else:
            return self.gather(np.argwhere(idx1), np.argwhere(idx2))
        
    
    def SplitBatches(self, Nbatches, FrameLinking=False):
        self.Nbatches=Nbatches
        if not FrameLinking:
            if self.linked:
                batches=np.random.randint(0,Nbatches,self.ch1.frame.shape[0])
                self.ch1.frame.assign(batches)
                self.ch2.frame.assign(batches)
                self.ch20linked.frame.assign(batches)
            else:
                batches1=np.random.randint(0,Nbatches,self.ch1.frame.shape[0])
                batches2=np.random.randint(0,Nbatches,self.ch2.frame.shape[0])
                self.ch1.frame.assign(batches1)
                self.ch2.frame.assign(batches2)
                self.ch20linked.frame.assign(batches2)
            if self.Neighbours:
                batches=np.random.randint(0,Nbatches,self.ch1NN.frame.shape[0])
                self.ch1NN.frame.assign(batches)
                self.ch2NN.frame.assign(batches)
                
        else: # keeps the positions that are in the same frame within the same frame
            frame1=np.zeros((self.ch1.frame).shape[0])
            frame2=np.zeros((self.ch2.frame).shape[0])
            frame20=np.zeros((self.ch20linked.frame).shape[0])
            if self.Neighbours:
                frameNN1=np.zeros((self.ch1NN.frame).shape[0])
                frameNN2=np.zeros((self.ch2NN.frame).shape[0])

            for frame in tf.unique(self.ch1.frame)[0]:
                batch=np.random.randint(0,Nbatches)                
                frame1[np.argwhere(self.ch1.frame==frame)]=batch
                frame2[np.argwhere(self.ch2.frame==frame)]=batch
                frame20[np.argwhere(self.ch20linked.frame==frame)]=batch
                if self.Neighbours:      
                    frameNN1[np.argwhere(self.ch1NN.frame==frame)]=batch
                    frameNN2[np.argwhere(self.ch2NN.frame==frame)]=batch
                    
            self.ch1.frame.assign(tf.Variable(frame1,dtype=tf.float32,trainable=False))
            self.ch2.frame.assign(tf.Variable(frame2,dtype=tf.float32,trainable=False))
            self.ch20linked.frame.assign(tf.Variable(frame20,dtype=tf.float32,trainable=False))
            if self.Neighbours:
                self.ch1NN.frame.assign(tf.Variable(frameNN1,dtype=tf.float32,trainable=False))
                self.ch2NN.frame.assign(tf.Variable(frameNN2,dtype=tf.float32,trainable=False))
                    
        
    def SplitDataset(self, linked=None):
    # Splits dataset into 2 halves for cross validation)
        if linked is None: linked=self.linked
        
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
    
    
    def SplitDatasetClusters(self, linked=None):
    # Splits dataset into 2 halves for cross validation)
        if linked is None: linked=self.linked
        
        N1=self.ch1.pos.shape[0]
        N2=self.ch2.pos.shape[0]
        if linked: raise Exception('Splitting clusters does not work when data is linked')
        clust1=self.ch1.ClusterCOM()
        clust2=self.ch2.ClusterCOM()
        idx1,idx2=self.link_clusters(clust1[0], clust1[1], clust2[0], clust2[1], maxDistance=5000)
        
        mask1=np.zeros(N1, dtype=bool)
        mask2=np.zeros(N2, dtype=bool)
        for i in range(len(idx1)):
            g=bool(np.random.randint(0,2))
            mask1[np.argwhere(self.ch1.group==idx1[i])]=g
            mask2[np.argwhere(self.ch2.group==idx2[i])]=g
        
        self.idx1=mask1
        self.idx2=mask2
        other1=self.gather(np.argwhere(mask1), np.argwhere(mask2) )
        other2=self.gather(np.argwhere((mask1-1).astype('bool')), np.argwhere((mask2-1).astype('bool')))
            
        return other1, other2
        
    
    def gather(self, idx1, idx2):
    # gathers the indexes of both Channels
        if len(idx1)==0 or len(idx2)==0: raise ValueError('Cannot gather a zeros sized array')
        other = copy.deepcopy(self)
        del other.ch1, other.ch2, other.ch20linked
        other.ch1 = Channel(pos=tf.gather_nd(self.ch1.pos,idx1), frame=tf.gather_nd(self.ch1.frame,idx1), 
                            group=tf.gather_nd(self.ch1.group,idx1))
        other.ch2 = Channel(pos=tf.gather_nd(self.ch2.pos,idx2), frame=tf.gather_nd(self.ch2.frame,idx2), 
                            group=tf.gather_nd(self.ch2.group,idx2))
        other.ch10 = Channel(pos=tf.gather_nd(self.ch10.pos,idx1), frame=tf.gather_nd(self.ch10.frame,idx1), 
                            group=tf.gather_nd(self.ch10.group,idx1))
        other.ch20 = Channel(pos=tf.gather_nd(self.ch20.pos,idx2), frame=tf.gather_nd(self.ch20.frame,idx2), 
                            group=tf.gather_nd(self.ch20.group,idx2))
        other.ch20linked = Channel(pos=tf.gather_nd(self.ch20linked.pos,idx2), frame=tf.gather_nd(self.ch20linked.frame,idx2), 
                            group=tf.gather_nd(self.ch20linked.group,idx2))
        return other
    
    
    #%% pair_functions
    def link_dataset(self, maxDistance=None,FrameLinking=None):
        if maxDistance is None: maxDistance=np.float32(1000)
        else: maxDistance=np.float32(maxDistance)
        try: #% linking datasets that are simulated works simpler
            self.relink_dataset()
        except:
            print('Linking Datasets for localizations within a distance of',maxDistance,'nm...')
            if self.linked: print('WARNING: Dataset already linked')
            if FrameLinking is None: FrameLinking=self.FrameLinking
            ch1_frame=self.ch1.frame.numpy()
            ch2_frame=self.ch2.frame.numpy()
            ch20_frame=self.ch20linked.frame.numpy()
            ch1_pos=self.ch1.pos.numpy()
            ch2_pos=self.ch2.pos.numpy()
            ch20_pos=self.ch20linked.pos.numpy()
            ch1_group=self.ch1.group.numpy()
            ch2_group=self.ch2.group.numpy()
            ch20_group=self.ch20linked.group.numpy()
            
            (pos1, frame1, pos2, frame2, pos20, frame20, group1, group2, group20) = ([],[],[],[],[],[],[],[],[])
            if FrameLinking: ## Linking per frame
                frame,_=tf.unique(self.ch1.frame)
                self.counts_linked=[]
                for fr in frame:
                    # Generate neighbouring indices per frame
                    framepos1=ch1_pos[ch1_frame==fr,:]
                    framepos2=ch2_pos[ch2_frame==fr,:]
                    framepos20=ch20_pos[ch20_frame==fr,:]
                    framegroup1=ch1_group[ch1_frame==fr]
                    framegroup2=ch2_group[ch2_frame==fr]
                    framegroup20=ch20_group[ch20_frame==fr]
                    
                    with Context() as ctx: # loading all NN
                        counts,indices = PostProcessMethods(ctx).FindNeighbors(framepos1, framepos2, maxDistance)
                
                    ## putting all idx in a list 
                    (idxlist, pos, i) = ([], 0,0)
                    for count in counts:
                        idxlist.append( np.stack([
                            i * np.ones([count], dtype=int),
                            indices[pos:pos+count] 
                            ]) )
                        pos+=count
                        i+=1
                    
                    for idx in idxlist:
                        if len(idx[0])!=0:
                            posA=framepos1[idx[0][0],:]
                            posB=framepos2[idx[1],:]
                            posB0=framepos20[idx[1],:]
                            ii=np.argmin( np.sum((posA-posB)**2) )
                            
                            frame1.append(fr)
                            frame2.append(fr)
                            frame20.append(fr)
                            pos1.append(posA)
                            pos2.append(posB[ ii ,:])
                            pos20.append(posB0[ ii ,:])
                            group1.append(framegroup1[idx[0][0]])
                            group2.append(framegroup2[idx[1]][ii])
                            group20.append(framegroup20[idx[1]][ii])
                            
                self.counts_linked.append(tf.reduce_sum(counts))
                
            else: ## taking the whole dataset as a single batch
                with Context() as ctx: # loading all NN
                    counts,indices = PostProcessMethods(ctx).FindNeighbors(ch1_pos, ch2_pos, maxDistance)
            
                ## putting all NNidx in a list 
                (idxlist, pos, i) = ([], 0,0)
                for count in counts:
                    idxlist.append( np.stack([
                        i * np.ones([count], dtype=int),
                        indices[pos:pos+count] 
                        ]) )
                    pos+=count
                    i+=1
                    
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
                        group1.append(ch1_group[i])
                        group2.append(ch2_group[idx[1][j]])
                        group20.append(ch20_group[idx[1][j]])
            
            if len(pos1)==0 or len(pos2)==0: raise ValueError('When Coupling Datasets, one or both of the Channels returns empty')
            
            del self.ch1, self.ch2, self.ch20linked
            self.ch1 = Channel( np.array(pos1) , np.array(frame1), np.array(group1) )
            self.ch2 = Channel( np.array(pos2) , np.array(frame2), np.array(group2) )
            self.ch20linked = Channel( np.array(pos20) , np.array(frame20), np.array(group20) )
            self.linked = True       
        
        
    def link_clusters(self, clusterpos1, clusterlist1, clusterpos2, clusterlist2, maxDistance=5000):
        with Context() as ctx: # loading all NN
            counts,indices = PostProcessMethods(ctx).FindNeighbors(clusterpos1, clusterpos2, maxDistance)
            
        pos,i=(0,0)
        clust1,clust2=([],[])
        for count in counts:
            if count!=0:
                pos1=tf.gather(clusterpos1, i * np.ones([count], dtype=int))
                pos2=tf.gather(clusterpos2, indices[pos:pos+count])
                lst1=tf.gather(clusterlist1, i * np.ones([count], dtype=int))
                lst2=tf.gather(clusterlist2, indices[pos:pos+count])
                indx=np.argmin( tf.reduce_sum(tf.pow(pos1-pos2,2), axis=1) )
                clust1.append(tf.gather(lst1,indx))
                clust2.append(tf.gather(lst2,indx))
            pos+=count
            i+=1
        return tf.stack(clust1), tf.stack(clust2)
        
        
    #%% Generate 
    def load_kNN(self, k, maxDistance=2000):
        self.ch1NN, self.ch2NN = self.kNearestNeighbour(self.ch1.pos.numpy(), self.ch2.pos.numpy(), 
                                                        k=k, maxDistance=maxDistance)
    
    
    def kNearestNeighbour(self, pos1=None, pos2=None, k=8, maxDistance=2000):
        if pos1 is None: pos1=self.ch1.pos.numpy()
        if pos2 is None: pos2=self.ch2.pos.numpy()
        with Context() as ctx: # loading all NN
            counts,indices = PostProcessMethods(ctx).FindNeighbors(pos1, pos2, maxDistance)
    
        pos,i=(0,0)
        pos1NN,pos2NN=([],[])
        for count in counts:
            if count!=0:
                pos1sbs=tf.gather(pos1, i * np.ones([count], dtype=int))
                pos2sbs=tf.gather(pos2, indices[pos:pos+count])
                indx=np.argsort( tf.reduce_sum(tf.pow(pos1sbs-pos2sbs,2), axis=1) )[:k]
                pos1NN.append(tf.gather(pos1sbs,indx))
                pos2NN.append(tf.gather(pos2sbs,indx))
            pos+=count
            i+=1
        self.ch1NN=tf.reshape(tf.concat(pos1NN,axis=0), [-1,2])
        self.ch2NN=tf.reshape(tf.concat(pos2NN,axis=0), [-1,2])
        return self.ch1NN, self.ch2NN
    
    
    def find_neighbours(self, maxDistance=50, FrameLinking=None):
    # Tries to generate neighbours according to all spots
        print('Finding neighbours within a distance of',maxDistance,'nm.')
        if FrameLinking is None: FrameLinking=self.FrameLinking
        maxDistance=np.float32(maxDistance)
        self.NN_maxDistance=maxDistance
        
        if FrameLinking:
            frame,_=tf.unique(self.ch1.frame)
            pos1, pos2, frame1=([],[],[])
            self.counts_Neighbours,self.Neighbours_mat=([],[])  ############
            for fr in frame:
                # Generate neighbouring indices per frame
                framepos1=self.ch1.pos.numpy()[self.ch1.frame==fr,:]
                framepos2=self.ch2.pos.numpy()[self.ch2.frame==fr,:]
                with Context() as ctx: # loading all NN
                    counts,indices = PostProcessMethods(ctx).FindNeighbors(framepos1, framepos2, maxDistance)
            
                pos,i=(0,0)
                for count in counts:
                    pos1.append(tf.gather(framepos1, i * np.ones([count], dtype=int)))
                    pos2.append(tf.gather(framepos2, indices[pos:pos+count]))
                    frame1.append( fr * np.ones([count], dtype=int) )
                    pos+=count
                    i+=1
                    
                self.counts_Neighbours.append(tf.reduce_sum(counts))
                #self.Neighbours_mat.append(self.GenerateNeighboursMat(framepos1, counts))
                
            #if not self.BatchOptimization: self.Neighbours_mat=self.AppendMat(self.Neighbours_mat)
        else:
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
            
            #self.Neighbours_mat=self.GenerateNeighboursMat(counts)
            
        # load matrix as Channel class
        self.ch1NN = Channel( tf.concat(pos1, axis=0), tf.concat(frame1, axis=0) )
        self.ch2NN = Channel( tf.concat(pos2, axis=0), tf.concat(frame1, axis=0) )
        self.Neighbours=True
        
        
    def random_choice(self,original_length, final_length):
        if original_length<final_length: raise ValueError('Invalid Input')
        lst=[]
        while len(lst)<final_length:
            r=np.random.randint(0,original_length)
            if r not in lst: lst.append(r)
        return lst
    
        
    #%% Filter
    def Filter(self, maxDistance):
    # The function for filtering both pairs and neigbhours 
        if self.linked: self.Filter_Pairs(maxDistance)
        #if self.Neighbours: self.Filter_Neighbours(maxDistance)
        
        
    def Filter_Pairs(self, maxDistance=150):
    # Filter pairs above maxDistance
        if maxDistance is not None:
            print('Filtering pairs above',maxDistance,'nm...')
            if not self.linked: raise Exception('Dataset should be linked before filtering pairs!')
            N0=self.ch1.pos.shape[0]
            
            dists = np.sqrt(np.sum( (self.ch1.pos.numpy() - self.ch2.pos.numpy())**2 , axis=1))
            idx = np.argwhere(dists<maxDistance)
            
            ch1_pos = self.ch1.pos.numpy()[idx[:,0],:]
            ch2_pos = self.ch2.pos.numpy()[idx[:,0],:]
            ch20_pos = self.ch20linked.pos.numpy()[idx[:,0],:]
            ch1_frame = self.ch1.frame.numpy()[idx[:,0]]
            ch2_frame = self.ch2.frame.numpy()[idx[:,0]]
            ch20_frame = self.ch20linked.frame.numpy()[idx[:,0]]
            
            if ch1_pos.shape[0]==0: raise Exception('All positions will be filtered out in current settings!')
            del self.ch1, self.ch2
            self.ch1 = Channel(ch1_pos, ch1_frame)
            self.ch2 = Channel(ch2_pos, ch2_frame)
            self.ch20linked = Channel(ch20_pos, ch20_frame)
            N1=self.ch1.pos.shape[0]
            print('Out of the '+str(N0)+' pairs localizations, '+str(N0-N1)+' have been filtered out ('+str(round((1-(N1/N0))*100,1))+'%)')
        else:
            print('Filtering is turned off, will pass without filtering.')
        