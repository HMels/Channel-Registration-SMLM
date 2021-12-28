
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
plt.rc('font', size=10)
import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from CatmullRomSpline2D import CatmullRomSpline2D
from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform


#%% training fn
def Train_Model(DS1, DS2, model, lr=1, epochs=100, opt_fn=tf.optimizers.Adagrad,
                    ch1=None, ch2=None, opt=None, maxDistance=1000):
        if epochs!=0 and epochs is not None:
            if DS1.BatchOptimization:
                if DS1.execute_linked: batches=DS1.counts_linked
                else: batches=DS1.counts_Neighbours
                if batches is None: raise Exception('Batches have not been initialized yet!')
                DS1.Nbatches=len(batches)
                print('Training '+model.name+' Mapping with (lr,#it)='+str((lr,epochs))+' for '+str(DS1.Nbatches)+' Batches...')
            else:
                print('Training '+model.name+' Mapping with (lr,#it)='+str((lr,epochs))+'...')
                batches=None                                             # take whole dataset as single batch
            
            ## Initialize Variables
            if ch1 is None and ch2 is None:
                if DS1.execute_linked and DS1.linked:
                    ch1, ch2 = DS1.ch1, DS1.ch2
                elif (not DS1.execute_linked):
                    ch1, ch2 = (DS1.ch1NN, DS1.ch2NN)
                elif DS1.execute_linked and (not DS1.linked): raise Exception('Tried to execute linked but dataset has not been linked.')
                elif (not DS1.execute_linked) and (not DS1.Neighbours): raise Exception('Tried to execute linked but no neighbours have been generated.')
                else:
                    raise Exception('Dataset is not linked but no Neighbours have been generated yet')
    
            ## The training loop
            if opt is None: opt=opt_fn(lr)
            loss_tot1=[]
            loss_tot2=[]
            for i in range(epochs):
                loss0=DS1.train_step(model, epochs, opt, ch1, ch2, batches)
                
                if i%10==0:
                    DS01=copy.deepcopy(DS1)
                    DS01.copy_models(DS1)
                    DS01.Apply_Splines()
                    
                    DS02=copy.deepcopy(DS2)
                    DS02.copy_models(DS1)
                    DS02.Apply_Splines()
                    
                    if DS1.execute_linked:
                        ch1_input=DS1.InputSplines(DS01.ch1.pos)
                        ch2_input=DS1.InputSplines(DS01.ch2.pos)
                        ch1_input1=DS1.InputSplines(DS02.ch1.pos)
                        ch2_input1=DS1.InputSplines(DS02.ch2.pos)
                        loss1=tf.reduce_sum(tf.square(ch1_input-ch2_input))
                        loss2=tf.reduce_sum(tf.square(ch1_input1-ch2_input1))
                    else:
                        ch1_input=DS1.InputSplines(DS01.ch1NN.pos)
                        ch2_input=DS1.InputSplines(DS01.ch2NN.pos)
                        ch1_input1=DS1.InputSplines(DS02.ch1NN.pos)
                        ch2_input1=DS1.InputSplines(DS02.ch2NN.pos)
                        loss1=-tf.reduce_sum(tf.exp(-1*tf.reduce_sum(tf.square(ch1_input-(ch2_input))/(1e6),axis=-1)))
                        loss2=-tf.reduce_sum(tf.exp(-1*tf.reduce_sum(tf.square(ch1_input1-(ch2_input1))/(1e6),axis=-1)))
                    loss_tot1.append(loss1.numpy())
                    loss_tot2.append(loss2.numpy())
                    print(loss0.numpy(), loss1.numpy(), loss2.numpy())
                if i%50==0 and i!=0: print('iteration='+str(i)+'/'+str(epochs))
            return loss_tot1, loss_tot2
        else:
            return None, None
        
        
def TrainSplines(DS1, DS2, lr, epochs, gridsize, edge_grids=1,
                 opt_fn=tf.optimizers.SGD, maxDistance=300, k=8):
    # initializing and training the model
    ch1_input,ch2_input=DS1.InitializeSplines(gridsize=gridsize, edge_grids=edge_grids,
                                               maxDistance=maxDistance, k=k)
    DS1.SplinesModel=CatmullRomSpline2D(DS1.ControlPoints)
    loss_tot1, loss_tot2=Train_Model(DS1,  DS2, DS1.SplinesModel, lr=learning_rate, 
                                     epochs=epochs, opt_fn=opt_fn,  ch1=ch1_input, ch2=ch2_input)  
    DS1.ControlPoints = DS1.SplinesModel.ControlPoints
    return loss_tot1, loss_tot2



#%% dataset
if True: #% Load FRET clusters
    output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_Misc/'
    maxDistance=300
    k=8
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
                   'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
                  pix_size=159, loc_error=10, mu=0, imgshape=[256,512], 
                  linked=False, FrameLinking=True, BatchOptimization=False, execute_linked=False)
    DS1.load_dataset_hdf5(align_rcc=True, transpose=False)
    DS1, DS2=DS1.SplitDatasetClusters()
    DS1clust=DS1.ClusterDataset(loc_error=None)
    DS1clust.execute_linked=True
    DS2clust=DS2.ClusterDataset(loc_error=None)
    DS1clust.link_dataset(maxDistance=maxDistance)
    DS2clust.link_dataset(maxDistance=maxDistance)
    
    ## optimization params
    learning_rate=5e-5
    epochs=1000
    pair_filter=[None, None, maxDistance]
    gridsize=7500
    
    #% aligning clusters
    DS1clust.AffineLLS(maxDistance, k)
    #DS2clust.Apply_Affine(DS1clust.AffineMat)
    #loss_tot1clust, loss_tot2clust=TrainSplines(DS1clust, DS2clust, learning_rate, None, gridsize, 
    #                                  edge_grids=1, opt_fn=tf.optimizers.SGD, 
    #                                  maxDistance=maxDistance, k=k)
    
    #DS1clust.Train_Splines(2e-3, None, gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD, 
    #                  maxDistance=maxDistance, k=k)
    #DS1clust.Apply_Splines()
    
    #% applying clusters
    DS1.copy_models(DS1clust) ## Copy all mapping parameters
    DS1.Apply_Affine(DS1clust.AffineMat)
    #if DS1.SplinesModel is not None: DS1.Apply_Splines()
    DS2.copy_models(DS1clust) ## Copy all mapping parameters
    DS2.Apply_Affine(DS1clust.AffineMat)
    #if DS2.SplinesModel is not None: DS2.Apply_Splines()
        
    #% linking dataset
    if not DS1.Neighbours: DS1.kNearestNeighbour(k=k, maxDistance=maxDistance)
    if not DS2.Neighbours: DS2.kNearestNeighbour(k=k, maxDistance=maxDistance)
    if not DS1.linked: DS1.link_dataset(maxDistance=maxDistance)
    if not DS2.linked: DS2.link_dataset(maxDistance=maxDistance)
    DS1.AffineLLS(maxDistance, k)
    DS2.Apply_Affine(DS1.AffineMat)
    DS1.Filter(pair_filter[0]) 
    DS2.Filter(pair_filter[0]) 

#if loss_tot1clust is not None:
#    np.savetxt(output_path+'DataOutput/loss_tot1clust.txt',loss_tot1clust)
#    np.savetxt(output_path+'DataOutput/loss_tot2clust.txt',loss_tot2clust)

#%% 
if epochs is not None:
    DS1.execute_linked=True
    DS2.execute_linked=True
    loss_tot1, loss_tot2=TrainSplines(DS1, DS2, learning_rate, epochs, gridsize, 
                                      edge_grids=1, opt_fn=tf.optimizers.SGD, 
                                      maxDistance=maxDistance, k=k)

np.savetxt(output_path+'DataOutput/loss_tot1.txt',loss_tot1)
np.savetxt(output_path+'DataOutput/loss_tot2.txt',loss_tot2)



#%% plotten
dpi=450
lw=2
#loss_tot1clust=np.loadtxt(output_path+'DataOutput/loss_tot1clust.txt')
#loss_tot2clust=np.loadtxt(output_path+'DataOutput/loss_tot2clust.txt')
loss_tot1=np.loadtxt(output_path+'DataOutput/loss_tot1.txt')
loss_tot2=np.loadtxt(output_path+'DataOutput/loss_tot2.txt')

fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)
#ax.set_yscale('log')
ax.set_ylabel('loss')

ax.plot(np.arange(0,epochs,10), loss_tot1, '--',lw=lw, label='localizations estimation')
ax.plot(np.arange(0,epochs,10), loss_tot2, '--',lw=lw, label='localizations testing')
ax.set_xlabel('iteration')
ax.legend(frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig(output_path+'Figures/loss_curve', transparent=True, dpi=dpi)