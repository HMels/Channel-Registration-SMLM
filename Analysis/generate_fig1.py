# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:46:39 2021

@author: Mels
"""

from matplotlib import lines, pyplot as plt
import tensorflow as tf
import copy
import numpy as np
import time
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
from Channel import Channel

from Align_Modules.Affine import AffineModel
from Align_Modules.Polynomial3 import Polynomial3Model
from Align_Modules.RigidBody import RigidBodyModel
from Align_Modules.Splines import CatmullRomSpline2D
from Align_Modules.Shift import ShiftModel
plt.rc('font', size=25)

def annotate_image(ax, text):
    return None
    #ax.text(-9000, 9000, text, ha='left', va='top',
    #        size=20, weight='bold')
    

plt.close('all')
#%% Functions 
def gen_channel(DS1, precision=10, heatmap=False):
# Generates the channels as matrix
    print('Generating Channels as matrix...')       

    # normalizing system
    locs1 = DS1.ch1.pos  / precision
    locs2 = DS1.ch2.pos  / precision
    locs20 = DS1.ch20linked.pos  / precision
    locs1original = DS1.ch10.pos  / precision
    locs2original = DS1.ch20.pos  / precision
    
    # calculate bounds of the system
    DS1.precision=precision
    DS1.bounds = np.array([[-8000, 8000],[-8000,8000]] ,dtype=float)/precision
    DS1.size_img = np.abs(np.round( (DS1.bounds[:,1] - DS1.bounds[:,0]) , 0).astype('int')    )        
    DS1.axis = np.array([ DS1.bounds[1,:], DS1.bounds[0,:]]) * DS1.precision
    DS1.axis = np.reshape(DS1.axis, [1,4])[0]
    
    # generating the matrices to be plotted
    DS1.channel1 = DS1.generate_matrix(locs1, heatmap)
    DS1.channel2 = DS1.generate_matrix(locs2, heatmap)
    DS1.channel20 = DS1.generate_matrix(locs20, heatmap)
    DS1.channel1original = DS1.generate_matrix(locs1original, heatmap)
    DS1.channel2original = DS1.generate_matrix(locs2original, heatmap)
    
    
    
def generate_matrix(DS1, locs, heatmap=False):
# takes the localizations and puts them in a channel
    channel = np.zeros([DS1.size_img[0]+1, DS1.size_img[1]+1], dtype = int)
    for i in range(locs.shape[0]):
        loc = np.round(locs[i,:],0).astype('int')
        if DS1.isin_domain(loc):
            loc -= np.round(DS1.bounds[:,0],0).astype('int') # place the zero point on the left
            if heatmap: channel[loc[0], loc[1]] += 1
            else: channel[loc[0], loc[1]] = 1
    return channel
    

def plt_channel(DS1, figsize=(12,6), annotate=None):
    print('Plotting...')
    gen_channel(DS1,precision=DS1.pix_size)
    channel1=np.flipud(DS1.channel1)
    channel2=np.flipud(DS1.channel2)
        
    # plotting all channels
    fig=plt.figure(figsize=figsize) 
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(channel1, extent = DS1.axis)
    #ax1.set_xlabel(label[0])
    #ax1.set_ylabel(label[1])
    ax1.set_xlim([-8000,8000])
    ax1.set_ylim([-8000,8000])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8)), s='channel 1')
    
    ax2.imshow(channel2, extent = DS1.axis)
    #ax2.set_xlabel(label[0])
    #ax2.set_ylabel(label[1])
    ax2.set_xlim([-8000,8000])
    ax2.set_ylim([-8000,8000])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8)), s='channel 2')
    
    if annotate is not None: annotate_image(ax1, annotate)
    plt.tight_layout()


def plt_linking(DS1, maxDistance=2000, annotate=None, figsize=(12,6)):
    def addCircle(ax1, ax2, idx, precision, maxDistance=2000, txt='', txtax=None):
        ax1.add_patch(plt.Circle( (DS1.ch10.pos[idx,1]-precision/2, DS1.ch10.pos[idx,0]+precision/2),
                                                   maxDistance,  fc='none', ec='red' ))
        ax2.add_patch(plt.Circle( (DS1.ch10.pos[idx,1]-precision, DS1.ch10.pos[idx,0]+precision),
                                                   maxDistance,  fc='none', ec='red' ))
        if txt!='':
            txtax.annotate(txt, (DS1.ch10.pos[idx,1]-maxDistance, 
                                 DS1.ch10.pos[idx,0]+precision*1.5), color='red')
       
        
    print('Plotting...')
    precision=DS1.pix_size
    gen_channel(DS1,precision=precision)
    channel1=np.flipud(DS1.channel1)
    channel2=np.flipud(DS1.channel20)
    
    # plotting all channels
    fig=plt.figure(figsize=figsize) 
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ## That point right bottom
    addCircle(ax1, ax2, -3, precision, maxDistance, '', ax2)
    ## Double linked point 
    addCircle(ax1, ax2, 15, precision, maxDistance, '', ax2)
    
    ax1.imshow(channel1, extent = DS1.axis)
    unique, unique_idx=tf.unique(DS1.ch1.frame[:])
    count=1
    idx_lst=np.argsort(DS1.ch2.frame[:])
    for i in range(idx_lst.shape[0]): # the linked pairs
        idx=idx_lst[i]
        frame=DS1.ch2.frame[idx]
        ax1.annotate(str(count), (DS1.ch1.pos[idx,1], DS1.ch1.pos[idx,0]), color='green')
        if frame!=DS1.ch2.frame[idx_lst[i-1]]:
            ax2.annotate(str(count), (DS1.ch2.pos[idx,1], DS1.ch2.pos[idx,0]), color='green')
            count+=1
            
    for i in range(DS1.ch10.pos.shape[0]): # the positions that will be deleted
        if (not DS1.ch10.pos[i,0] in DS1.ch1.pos[:,0]) and (not DS1.ch10.pos[i,1] in DS1.ch1.pos[:,1]):
            ax1.plot(DS1.ch10.pos[i,1]-precision/2, DS1.ch10.pos[i,0]+precision/2, 'x', color='yellow')
            addCircle(ax1, ax2, i, precision, maxDistance, '', ax1)
            
    #ax1.set_xlabel(label[0])
    #ax1.set_ylabel(label[1])
    ax1.set_xlim([-8000,8000])
    ax1.set_ylim([-8000,8000])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8)), s='channel 1')
    
    ax2.imshow(channel2, extent = DS1.axis)
    for i in range(DS1.ch20.pos.shape[0]): # the positions that will be deleted
        if (not DS1.ch20.pos[i,0] in DS1.ch2.pos[:,0]) and (not DS1.ch20.pos[i,1] in DS1.ch2.pos[:,1]):
            ax2.plot(DS1.ch20.pos[i,1]-precision/2, DS1.ch20.pos[i,0]+precision/2, 'x', color='yellow')
            #ax2.scatter( DS1.ch20.pos[i,1]-precision/2 , DS1.ch20.pos[i,0]+precision/2 , s=maxDistance,  facecolors='none', edgecolors='red' )
            #ax1.scatter( DS1.ch20.pos[i,1]-precision/2 , DS1.ch20.pos[i,0]+precision/2 , s=maxDistance,  facecolors='none', edgecolors='red' )
    
    #ax2.set_xlabel(label[0])
    #ax2.set_ylabel(label[1])
    ax2.set_xlim([-8000,8000])
    ax2.set_ylim([-8000,8000])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8)), s='channel 2')
    
    if annotate is not None: annotate_image(ax1, annotate)
    plt.tight_layout()
    plt.show() 
    
    
def plt_filter(DS1, annotate=None, figsize=(12,6), shift=None):
    print('Plotting...')
    precision=DS1.pix_size
    gen_channel(DS1,precision=precision)
    channel1=np.flipud(DS1.channel1)
    channel2=np.flipud(DS1.channel2)
        
    # plotting all channels
    fig=plt.figure(figsize=figsize) 
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(channel1, extent = DS1.axis)
    #ax1.set_xlabel(label[0])
    #ax1.set_ylabel(label[1])
    ax1.set_xlim([-8000,8000])
    ax1.set_ylim([-8000,8000])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                 fc=(1., 0.8, 0.8)), s='channel 1')
    
    ax2.imshow(channel2, extent = DS1.axis)
    #ax2.set_xlabel(label[0])
    #ax2.set_ylabel(label[1])
    ax2.set_xlim([-8000,8000])
    ax2.set_ylim([-8000,8000])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                 fc=(1., 0.8, 0.8)), s='mapped channel 2')
    
    if shift is not None:
        ax2.arrow(7500, 7500, shift[1], 0,
                  width=.5, length_includes_head=False, facecolor='red', edgecolor='red', head_width=100)
        ax2.arrow(7500, 7500, 0, shift[0],
                  width=.5, length_includes_head=False, facecolor='red', edgecolor='red', head_width=100)
        for i in range(DS1.ch20.pos.shape[0]):
            ax2.arrow(DS1.ch20.pos[i,1]-shift[1],DS1.ch20.pos[i,0]-shift[0],shift[1],shift[0], width=.2, 
                      length_includes_head=True, facecolor='red', edgecolor='red', head_width=100)
    
    if annotate is not None: annotate_image(ax1, annotate)
    fig.tight_layout()
    
    
    for i in range(DS1.ch20.pos.shape[0]): # the positions that will be deleted
        if (not DS1.ch20.pos[i,0] in DS1.ch2.pos[:,0]) and (not DS1.ch20.pos[i,1] in DS1.ch2.pos[:,1]):
            ax2.plot(DS1.ch20.pos[i,1]-precision/2, DS1.ch20.pos[i,0]+precision/2, 'x', color='yellow')
    for i in range(DS1.ch10.pos.shape[0]): # the positions that will be deleted
        if (not DS1.ch10.pos[i,0] in DS1.ch1.pos[:,0]) and (not DS1.ch10.pos[i,1] in DS1.ch1.pos[:,1]):
            ax1.plot(DS1.ch10.pos[i,1]-precision/2, DS1.ch10.pos[i,0]+precision/2, 'x', color='yellow')
    
    
def plt_grid(DS1, locs_markersize=25, d_grid=.1, Ngrids=1, plotarrows=True, plotmap=False, annotate=None, figsize=(12,6)):
    print('Plotting...')
    gen_channel(DS1,precision=DS1.pix_size)
    channel20=np.flipud(DS1.channel20)
    channel2=np.flipud(DS1.channel2)
    #label=['x2', 'x1']
    
    
    ## Horizontal Grid
    #ControlPoints=tf.stack([
    #    DS1.ControlPoints[DS1.edge_grids:-DS1.edge_grids-1,DS1.edge_grids:-DS1.edge_grids-1,0],
    #    DS1.ControlPoints[DS1.edge_grids:-DS1.edge_grids-1,DS1.edge_grids:-DS1.edge_grids-1,1]
    #    ], axis=-1)
    #Hx1_grid = tf.range(0, tf.reduce_max(ControlPoints[:,:,0])+d_grid, delta=d_grid, dtype=tf.float32)
    #Hx2_grid = tf.range(0, tf.reduce_max(ControlPoints[:,:,1])+d_grid, delta=1/Ngrids, dtype=tf.float32)
    Hx1_grid = tf.range(-8000/DS1.gridsize, 8000/DS1.gridsize, delta=d_grid, dtype=tf.float32)*DS1.gridsize
    Hx2_grid = tf.range(DS1.x1_min-10, 8000/DS1.gridsize, delta=1/Ngrids, dtype=tf.float32)*DS1.gridsize
    HGrid = tf.reshape(tf.stack(tf.meshgrid(Hx1_grid, Hx2_grid), axis=-1) , (-1,2)) 

    ## Vertical Grid
    #Vx1_grid = tf.range(0, tf.reduce_max(ControlPoints[:,:,0])+d_grid, delta=1/Ngrids, dtype=tf.float32)
    #Vx2_grid = tf.range(0, tf.reduce_max(ControlPoints[:,:,1])+d_grid, delta=d_grid, dtype=tf.float32)
    Vx1_grid = tf.range(DS1.x2_min-10, 8000/DS1.gridsize, delta=1/Ngrids, dtype=tf.float32)*DS1.gridsize
    Vx2_grid = tf.range(-8000/DS1.gridsize, 8000/DS1.gridsize, delta=d_grid, dtype=tf.float32)*DS1.gridsize
    VGrid = tf.gather(tf.reshape(tf.stack(tf.meshgrid(Vx2_grid, Vx1_grid), axis=-1) , (-1,2)), [1,0], axis=1)
    
    
    fig=plt.figure(figsize=figsize) 
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(channel20, extent = DS1.axis)
    gridmapping(ax1, DS1, HGrid, VGrid, Hx1_grid, Vx2_grid, Ngrids, plotarrows=False)
    #ax1.set_xlabel(label[0])
    #ax1.set_ylabel(label[1])
    ax1.set_xlim([-8000,8000])
    ax1.set_ylim([-8000,8000])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8)), s='original channel 2')
    plt.tight_layout()
    
    HGrid = DS1.InputSplines(DS1.SplinesModel( DS1.InputSplines(HGrid) ), inverse=True)
    VGrid = DS1.InputSplines(DS1.SplinesModel( DS1.InputSplines(VGrid) ), inverse=True)
    
    ax2.imshow(channel2, extent = DS1.axis)
    gridmapping(ax2, DS1, HGrid, VGrid, Hx1_grid, Vx2_grid, Ngrids, plotarrows=True)
    #ax2.set_xlabel(label[0])
    #ax2.set_ylabel(label[1])
    ax2.set_xlim([-8000,8000])
    ax2.set_ylim([-8000,8000])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(x=-7750,y=7750, va='top', bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8)), s='mapped channel 2')
    
    if annotate is not None: annotate_image(ax1, annotate)
    plt.tight_layout()
    
    
def gridmapping(ax, DS1, HGrid, VGrid, Hx1_grid, Vx2_grid, Ngrids, plotarrows=True, CP_markersize=20):
    
    # plotting the localizations
    if plotarrows:
        for i in range(DS1.ch1.pos.shape[0]):
            ax.arrow(DS1.ch20linked.pos[i,1],DS1.ch20linked.pos[i,0], DS1.ch2.pos[i,1]-DS1.ch20linked.pos[i,1],
                      DS1.ch2.pos[i,0]-DS1.ch20linked.pos[i,0], width=.2, 
                      length_includes_head=True, facecolor='red', edgecolor='red', head_width=100)
    
    # plotting the ControlPoints
    #plt.scatter(ControlPoints[:,:,0]*DS1.gridsize, ControlPoints[:,:,1]*DS1.gridsize,
    #            c='b', marker='d', s=CP_markersize, label='ControlPoints')
    
    (nn, i,j)=(Hx1_grid.shape[0],0,0)
    while i<HGrid.shape[0]:
        if j%Ngrids==0:
            ax.plot(HGrid[i:i+nn,0], HGrid[i:i+nn,1], c='c')
        else:
            ax.plot(HGrid[i:i+nn,0], HGrid[i:i+nn,1], c='b')
        i+=nn
        j+=1
        
    (nn, i,j)=(Vx2_grid.shape[0],0,0)
    while i<VGrid.shape[0]:
        if j%Ngrids==0:
            ax.plot(VGrid[i:i+nn,0], VGrid[i:i+nn,1], c='c')
        else:
            ax.plot(VGrid[i:i+nn,0], VGrid[i:i+nn,1], c='b')
        i+=nn
        j+=1
        
        
def copy_image(DS1):
    DS1.ch10=copy.deepcopy(DS1.ch1)
    DS1.ch20=copy.deepcopy(DS1.ch2)
    DS1.ch20linked=copy.deepcopy(DS1.ch2)
    return DS1
        
#%%
DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
              pix_size=159, loc_error=10, mu=0.3,
              linked=False, FrameLinking=False, BatchOptimization=False)

pos=np.array([
    [5000, -2000], # left eye
    #[4000, -2000],
    [3000, -2000],
    #[2000, -2000],
    [1000, -2000],
    [5000, 2000], # right eye
    #[4000, 2000],
    [3000, 2000],
    #[2000, 2000],
    [1000, 2000], # mouth (left part)
    [1000, -6000],
    [0, -5200],
    [-1000, -4700],
    [-2000, -3600],
    [-3000, -2600],
    [-4000, -1500],
    [-4300, -500], # mouth (right part)
    [1000, 6000],
    [0, 5200],
    [-1000, 4700],
    [-2000, 3600],
    [-3000, 2600],
    [-4000, 1500],
    [-4300, 500],    
    ])

pos1=np.concatenate((pos,np.array([[-4000, -4000], [-7000, 7000], [7000, -3000], [6000, 7000]])), axis=0)
pos2=np.concatenate((pos,np.array([[-6000, -800], [-6000,-6000], [-6000, 5200], [2000, -7000]])))

np.random.seed(1)
DS1.ch1=Channel(tf.Variable(pos1+DS1.loc_error*np.random.randn(pos1.shape[0],2),dtype=tf.float32, trainable=False), np.arange(pos1.shape[0]))
DS1.ch2=Channel(tf.Variable(pos2+DS1.loc_error*np.random.randn(pos2.shape[0],2),dtype=tf.float32, trainable=False), np.arange(pos2.shape[0]))
DS1.ch20linked=copy.deepcopy(DS1.ch2)
DS1.ch20=copy.deepcopy(DS1.ch2)
DS1.ch10=copy.deepcopy(DS1.ch1)

deform=Deform(shift=np.array([500,700]), rotation=5,shear=np.array([0.03, 0.02]), scaling=np.array([1.04,1.03 ]))
DS1.ch2.pos.assign(deform.deform(DS1.ch2.pos))

#%%
# original
plt_channel(DS1, annotate='A')
DS1=copy_image(DS1)

# linking
DS1.link_dataset(maxDistance=2000)
plt_linking(DS1, annotate='B')
DS1=copy_image(DS1)

# shift
DS1.ShiftModel=ShiftModel()
DS1.Train_Model(DS1.ShiftModel, lr=100, epochs=100, opt_fn=tf.optimizers.Adagrad)
DS1.Transform_Model(DS1.ShiftModel)
#plt_shift(DS1, annotate='C', figsize=(4.8,4.8))
DS1=copy_image(DS1)

# filter
DS1.Filter(1500) 
plt_filter(DS1, annotate='D', shift=DS1.ShiftModel.d.numpy())
DS1=copy_image(DS1)
    
# splines
ch1_input,ch2_input=DS1.InitializeSplines(gridsize=1500, edge_grids=1)
DS1.SplinesModel=CatmullRomSpline2D(DS1.ControlPoints)
DS1.Train_Model(DS1.SplinesModel, lr=1e-2, epochs=300, opt_fn=tf.optimizers.SGD, 
                 ch1=ch1_input, ch2=ch2_input)  
# applying the model
DS1.ControlPoints = DS1.SplinesModel.ControlPoints
DS1.ch2.pos.assign(DS1.InputSplines(
    DS1.Transform_Model(DS1.SplinesModel, ch2=DS1.InputSplines(DS1.ch2.pos)),
    inverse=True))
plt_grid(DS1, annotate='E')
DS1=copy_image(DS1)


# filter
DS1.Filter(200) 
plt_filter(DS1, annotate='F',figsize=(12,6))

#%% last image
plt_channel(DS1, annotate='G',figsize=(12,6))
'''
# reload points
np.random.seed(1)
DS1.ch1=Channel(tf.Variable(pos1+DS1.loc_error*np.random.randn(pos1.shape[0],2),dtype=tf.float32, trainable=False), np.arange(pos1.shape[0]))
DS1.ch2=Channel(tf.Variable(pos2+DS1.loc_error*np.random.randn(pos2.shape[0],2),dtype=tf.float32, trainable=False), np.arange(pos2.shape[0]))
DS1.ch20linked=copy.deepcopy(DS1.ch2)
DS1.ch20=copy.deepcopy(DS1.ch2)
DS1.ch10=copy.deepcopy(DS1.ch1)
deform=Deform(shift=np.array([500,700]), rotation=5,shear=np.array([0.03, 0.02]), scaling=np.array([1.04,1.03 ]))
DS1.ch2.pos.assign(deform.deform(DS1.ch2.pos))

# transform and plot
DS1.Transform_Model(DS1.ShiftModel)
DS1.ch2.pos.assign(DS1.InputSplines(
    DS1.Transform_Model(DS1.SplinesModel, ch2=DS1.InputSplines(DS1.ch2.pos)),
    inverse=True))
plt_channel(DS1, annotate='H')
'''