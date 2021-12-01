# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:36:49 2021

@author: Mels
"""
    
from matplotlib import lines, pyplot as plt
import tensorflow as tf
import copy
import time
import numpy as np
from scipy.optimize import curve_fit
import scipy.special as scpspc
import pandas as pd

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
from Channel import Channel
plt.rc('font', size=17)

def annotate_image(ax, text, displacement=[-1000,1000]):
    return None
    #ax.text(ax.get_xbound()[0]+displacement[0], ax.get_ybound()[1]+displacement[1], text, ha='left', va='top',
    #        size=20, weight='bold')
    
    
#%% Load HEL1.hdf5
maxDistance=1000
DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5', 
               'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5'],
              pix_size=159, loc_error=10, mu=0, coloc_error=np.sqrt(2)*(10),
              imgshape=[256,512], linked=False, FrameLinking=True, BatchOptimization=False)
DS1.load_dataset_hdf5(align_rcc=False)
DS1, DS2=DS1.SplitDataset()   

## optimization params
execute_linked=True
learning_rates = [1000, .1, 2e-3]
epochs = [100, None, 300]
pair_filter = [250, 250, 250]
gridsize=100
nbins=100
    

plt.close('all')
#%% fig1ab
def plot_channel(DS1, channel1, channel2):
    channel1=np.rot90(channel1)
    channel2=np.rot90(channel2)
    axis=np.concatenate((DS1.axis[2:],DS1.axis[:2]))
    label=['y-position [\u03bcm]','x-position [\u03bcm]']
    
    # plotting all channels
    fig=plt.figure(figsize=(12,11)) 
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax=[ax1,ax2]
    ax[0].imshow(channel1, extent = axis/1000)
    ax[0].set_xlabel(label[0])
    ax[0].set_ylabel(label[1])
    ax[0].set_title('channel 1')
    
    ax[1].imshow(channel2, extent = axis/1000)
    ax[1].set_xlabel(label[0])
    ax[1].set_yticks([])
    ax[1].set_title('channel 2')
    fig.tight_layout()
    return fig, ax
        

def plot_frame(DS1, frame=None, annotate=None, label=False):
    temp=copy.deepcopy(DS1)
    if frame is not None:
        framepos1=tf.gather_nd(temp.ch1.pos.numpy(),np.argwhere(temp.ch1.frame==frame))
        framepos2=tf.gather_nd(temp.ch2.pos.numpy(),np.argwhere(temp.ch2.frame==frame))
        framepos=tf.concat([framepos1, framepos2], axis=0)
        del temp.ch10, 
        temp.ch10=Channel(framepos, frame*np.ones(framepos.shape[0],dtype=float))
    
    temp.generate_channel(precision=temp.pix_size)
    channel1=np.flipud(temp.channel1)
    channel2=np.flipud(temp.channel2)
    fig, ax=plot_channel(temp, channel1, channel2)
    #fig, ax=temp.plot_1channel(channel1, figsize=(12,6), title='')
    
    if annotate is not None: annotate_image(ax[0], annotate, displacement=[-6000, 1000])
    if label and frame is not None: 
        if isinstance(label, list):
            for lbl in label:
                ax[0].annotate('1', (framepos1[lbl,1], framepos1[lbl,0]), color='red')
                ax[0].annotate('2', (framepos2[lbl,1]-1000, framepos2[lbl,0]), color='red')
        if isinstance(label, int):
            ax[0].annotate('1', (framepos1[label,1], framepos1[label,0]), color='red')
            ax[0].annotate('2', (framepos2[label,1]-1000, framepos2[label,0]), color='red')
            
    return fig,ax
    
plot_frame(DS1,None, annotate='A') ## fig 1a

#%%
DS1.link_dataset(maxDistance=maxDistance)
DS2.link_dataset(maxDistance=maxDistance)


#%% fig 1c
def ErrorDistribution_r(DS1, fig=None,ax=None, nbins=30, xlim=31, error=None, mu=None, fit_data=True, annotate=None, annotate_dist=[-10,0]):
    if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    if mu is None: mu=0
    pos1=DS1.ch1.pos_all()
    pos2=DS1.ch2.pos_all()
        
    # Calculating the error
    dist, avg, r = DS1.ErrorDist(pos1, pos2)
    dist=dist[np.argwhere(dist<xlim)]
    
    # plotting the histogram
    if fig is None: fig=plt.figure(figsize=(12,4)) 
    if ax is None: ax = fig.add_subplot(111)
    
    n=ax.hist(dist, range=[0,xlim], label='N='+str(pos1.shape[0]), alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
    #plt.axvline(x=avg, label='average='+str(round(avg,2))+'[nm]')
    ymax = np.max(n[0])*1.1
    
    
    if fit_data: ## fit bar plot data using curve_fit
        def func(r, sigma, mu=mu):
            # from Churchman et al 2006
            sigma2=sigma**2
            if mu==0:
                return r/sigma2*np.exp(-r**2/2/sigma2)
            else:
                return (r/sigma2)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)

        N = pos1.shape[0] * ( n[1][1]-n[1][0] )
        xn=(n[1][:-1]+n[1][1:])/2 
        popt, pcov = curve_fit(func, xn, n[0]/N, p0=np.std(xn))
        x = np.linspace(0, xlim, 1000)
        y = func(x, *popt)*N
        ax.plot(x, y, c='g',label=(r'fit: $\mu$='+str(round(mu,2))+', $\sigma$='+str(round(popt[0],2))+'nm'))
    
        ## plot how function should look like
        if error is not None:
            sgm=error
            y = func(x, sgm, mu)*N
            ax.plot(x, y, c='b',label=(r'optimum: $\mu$='+str(round(mu,2))+', $\sigma$='+str(round(sgm,2))+'nm'))
            if np.max(y)>ymax: ymax=np.max(y)*1.1

    # Some extra plotting parameters
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xlim])
    ax.set_xlabel('Absolute error [nm]')
    ax.set_ylabel('# of localizations')
    ax.legend(loc='upper right')
    fig.tight_layout()
    if annotate is not None: annotate_image(ax, annotate, displacement=annotate_dist)
    return fig, ax
    
## fig 1b
ErrorDistribution_r(DS2, nbins=nbins, xlim=maxDistance, error=None, mu=None, fit_data=False, annotate='B', annotate_dist=[-30,10])


#%% running the model
DS1.TrainRegistration(execute_linked=execute_linked, learning_rates=learning_rates, 
                      epochs=epochs, pair_filter=pair_filter, gridsize=gridsize)

DS1.reload_dataset()
DS1.ApplyRegistration()
DS1.link_dataset(maxDistance=maxDistance,FrameLinking=True)  

#%% fig1c
def plt_grid(DS1, fig=None,ax=None, locs_markersize=25, d_grid=.1, Ngrids=1, plotmap=False, annotate=None, window=None):
    print('Plotting...')
    def gen_channel(DS1, precision=10, heatmap=False):
        # normalizing system
        locs1 = DS1.ch1.pos  / precision
        locs2 = DS1.ch2.pos  / precision
        locs20 = DS1.ch20linked.pos  / precision
        locs1original = DS1.ch10.pos  / precision
        locs2original = DS1.ch20.pos  / precision
        # calculate bounds of the system
        DS1.precision=precision
        DS1.bounds = np.array([[-DS1.imgshape[1]/2*DS1.pix_size, DS1.imgshape[1]/2*DS1.pix_size]
                               ,[-DS1.imgshape[0]/2*DS1.pix_size,DS1.imgshape[0]/2*DS1.pix_size]] ,dtype=float)/precision
        DS1.size_img = np.abs(np.round( (DS1.bounds[:,1] - DS1.bounds[:,0]) , 0).astype('int')    )        
        DS1.axis = np.array([ DS1.bounds[1,:], DS1.bounds[0,:]]) * DS1.precision
        DS1.axis = np.reshape(DS1.axis, [1,4])[0]
        # generating the matrices to be plotted
        DS1.channel1 = DS1.generate_matrix(locs1, heatmap)
        DS1.channel2 = DS1.generate_matrix(locs2, heatmap)
        DS1.channel20 = DS1.generate_matrix(locs20, heatmap)
        DS1.channel1original = DS1.generate_matrix(locs1original, heatmap)
        DS1.channel2original = DS1.generate_matrix(locs2original, heatmap)
    
    
    def gridmapping(ax, DS1, d_grid, Ngrids, plotarrows=False, CP_markersize=20):
        ## Horizontal Grid
        Hx1_grid = tf.range(DS1.x1_min, DS1.x1_max, delta=d_grid, dtype=tf.float32)*DS1.gridsize
        Hx2_grid = tf.range(DS1.x2_min, DS1.x2_max, delta=1/Ngrids, dtype=tf.float32)*DS1.gridsize
        HGrid = tf.Variable( tf.reshape(tf.stack(tf.meshgrid(Hx1_grid, Hx2_grid), axis=-1) , (-1,2)) , trainable=False, dtype=tf.float32)
        ## Vertical Grid
        Vx1_grid = tf.range(DS1.x1_min, DS1.x1_max, delta=1/Ngrids, dtype=tf.float32)*DS1.gridsize
        Vx2_grid = tf.range(DS1.x2_min, DS1.x2_max, delta=d_grid, dtype=tf.float32)*DS1.gridsize
        VGrid = tf.Variable(tf.gather(tf.reshape(tf.stack(tf.meshgrid(Vx2_grid, Vx1_grid), axis=-1) , (-1,2)), [1,0], axis=1), trainable=False, dtype=tf.float32)
        # map the grids
        HGrid = DS1.InputSplines(DS1.SplinesModel( DS1.InputSplines(HGrid) ), inverse=True)
        VGrid = DS1.InputSplines(DS1.SplinesModel( DS1.InputSplines(VGrid) ), inverse=True)
        if plotarrows:
            for i in range(DS1.ch1.pos.shape[0]):
                ax.arrow(DS1.ch20linked.pos[i,1],DS1.ch20linked.pos[i,0], DS1.ch2.pos[i,1]-DS1.ch20linked.pos[i,1],
                          DS1.ch2.pos[i,0]-DS1.ch20linked.pos[i,0], width=.2, 
                          length_includes_head=False, facecolor='red', edgecolor='red', head_width=1)
                
        (nn, i,j)=(Hx1_grid.shape[0],0,0)
        while i<HGrid.shape[0]:
            if j%Ngrids==0:
                ax.plot(HGrid[i:i+nn,1], HGrid[i:i+nn,0], c='c')
            else:
                ax.plot(HGrid[i:i+nn,1], HGrid[i:i+nn,0], c='b')
            i+=nn
            j+=1
        (nn, i,j)=(Vx2_grid.shape[0],0,0)
        while i<VGrid.shape[0]:
            if j%Ngrids==0:
                ax.plot(VGrid[i:i+nn,1], VGrid[i:i+nn,0], c='c')
            else:
                ax.plot(VGrid[i:i+nn,1], VGrid[i:i+nn,0], c='b')
            i+=nn
            j+=1
            
    #fig=plt.figure(figsize=(19,5)) 
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    fig1=plt.figure(figsize=(12,6)) 
    fig2=plt.figure(figsize=(4,4))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111) 
    
    ##complete figure      
    DS1.generate_channel(precision=DS1.pix_size)
    channel2=np.flipud(DS1.channel2)
    channel2=DS1.channel2
    label=['x-position [\u03bcm]', 'y-position [\u03bcm]']    
        
    ax1.imshow(channel2, extent = DS1.axis)
    gridmapping(ax1, DS1, d_grid, Ngrids=1, plotarrows=False)
    ax1.set_xlabel(label[0])
    ax1.set_ylabel(label[1])
    ax1.set_xlim([DS1.x2_min*DS1.gridsize, DS1.x2_max*DS1.gridsize-DS1.pix_size])
    ax1.set_ylim([DS1.x1_min*DS1.gridsize, DS1.x1_max*DS1.gridsize-DS1.pix_size])
    
    ##zoom
    temp=copy.deepcopy(DS1)
    if window is None: window=np.array([[-20*temp.gridsize, -4*temp.gridsize],[-36*temp.gridsize, -20*temp.gridsize]])
    
    idx=(np.where(temp.ch2.pos.numpy()[:,1]>window[0,0],True,False)*
         np.where(temp.ch2.pos.numpy()[:,1]<window[0,1],True,False)*
         np.where(temp.ch2.pos.numpy()[:,0]>window[1,0],True,False)*
         np.where(temp.ch2.pos.numpy()[:,0]<window[1,1],True,False))
    pos1=temp.ch1.pos.numpy()[idx,:]
    pos2=temp.ch2.pos.numpy()[idx,:]
    del temp.ch1,temp.ch2
    temp.ch1=Channel(pos1, np.ones(pos1.shape[0]))
    temp.ch2=Channel(pos2, np.ones(pos2.shape[0]))
    
    temp.x1_min=window[0,0]/temp.gridsize
    temp.x1_max=window[0,1]/temp.gridsize
    temp.x2_min=window[1,0]/temp.gridsize
    temp.x2_max=window[1,1]/temp.gridsize
    temp.generate_channel(precision=temp.pix_size/10)

    channel2=np.flipud(temp.channel2)
    ax2.imshow(channel2, extent = temp.axis)
    gridmapping(ax2, DS1, d_grid/Ngrids, Ngrids=Ngrids, plotarrows=False)
    #ax2.set_xlabel(label[0])
    #ax2.set_ylabel(label[1])
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([temp.x1_min*temp.gridsize, temp.x1_max*temp.gridsize-temp.pix_size])
    ax2.set_ylim([temp.x2_min*temp.gridsize, temp.x2_max*temp.gridsize-temp.pix_size])
    
    ## shift arrows 
    ax2.arrow(-600,-3500, temp.ShiftModel.trainable_variables[0][1], 0,
              width=5, length_includes_head=True, facecolor='red', edgecolor='red', head_width=20)
    ax2.arrow(-600,-3500, 0, temp.ShiftModel.trainable_variables[0][0],
              width=5, length_includes_head=True, facecolor='red', edgecolor='red', head_width=20)
    
    if annotate is not None: annotate_image(ax1, annotate, displacement=[-4000,0])
    plt.tight_layout()
    
    ## draw zoomwindow 
    ax1.add_patch(plt.Rectangle((window[1,0],window[0,0]), window[1,1]-window[1,0], window[0,1]-window[0,0], ec='red', fc='none', linewidth=3.5, zorder=10))
    ax1.set_xlim([-10000,10000])
    ax1.set_ylim([-5000,5000])
    ax1.set_xticks([-10000,-5000,0,5000,10000])
    ax1.set_xticklabels([-10,-5,0,5,10])
    ax1.set_yticks([-5000,0,5000])
    ax1.set_yticklabels([-5,0,5])
    
    
plt_grid(DS1, locs_markersize=25, d_grid=.1, Ngrids=1, plotmap=False, window=None, annotate='C') ## fig 1c


#%% fig 1d
fig=plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

fig, ax1=ErrorDistribution_r(DS1, fig, ax1, nbins=nbins, xlim=pair_filter[2], 
                             error=DS1.coloc_error, mu=DS1.mu, fit_data=True, annotate='D', annotate_dist=[-40,0])


#%%
DS2.reload_dataset()
DS2.copy_models(DS1) ## Copy all mapping parameters
DS2.ApplyRegistration()
if not DS2.linked: 
    DS2.link_dataset(maxDistance=maxDistance,FrameLinking=True)   
DS2.Filter(pair_filter[1])
fig, ax2=ErrorDistribution_r(DS2, fig, ax2, nbins=nbins, xlim=pair_filter[2],
                             error=DS2.coloc_error, mu=DS2.mu, fit_data=True)


#%% fig1ef
def ErrorDistribution_xy(DS1, nbins=30, xlim=31, error=None, mu=None, fit_data=True, annotate=None):
        if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        if mu is None: mu=0
        pos1=DS1.ch1.pos_all()
        pos2=DS1.ch2.pos_all()
            
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        distx=pos1[:,0]-pos2[:,0]
        disty=pos1[:,1]-pos2[:,1]
        mask=np.where(distx<xlim,True, False)*np.where(disty<xlim,True,False)
        distx=distx[mask]
        disty=disty[mask]
        nx = ax[0].hist(distx, range=[-xlim,xlim], label='N='+str(pos1.shape[0]),alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        ny = ax[1].hist(disty, range=[-xlim,xlim], label='N='+str(pos1.shape[0]),alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        
        
        if fit_data: ## fit bar plot data using curve_fit
            def func(r, mu, sigma):
                return np.exp(-(r - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2*np.pi)*sigma)
            
            Nx = pos1.shape[0] * ( nx[1][1]-nx[1][0] )
            Ny = pos1.shape[0] * ( ny[1][1]-ny[1][0] )
            xn=(nx[1][:-1]+nx[1][1:])/2
            yn=(ny[1][:-1]+ny[1][1:])/2
            poptx, pcovx = curve_fit(func, xn, nx[0]/Nx, p0=[np.average(distx), np.std(distx)])
            popty, pcovy = curve_fit(func, yn, ny[0]/Ny, p0=[np.average(disty), np.std(disty)])
            x = np.linspace(-xlim, xlim, 1000)
            yx = func(x, *poptx)*Nx
            yy = func(x, *popty)*Ny
            ax[0].plot(x, yx, c='g',label=(r'fit: $\mu$='+str(round(poptx[0],2))+', $\sigma$='+str(round(poptx[1],2))+'nm'))
            ax[1].plot(x, yy, c='g',label=(r'fit: $\mu$='+str(round(popty[0],2))+', $\sigma$='+str(round(popty[1],2))+'nm'))
            ymax = np.max([np.max(nx[0]),np.max(ny[0]), np.max(yx), np.max(yy)])*1.1
            
            ## plot how function should look like
            if error is not None:
                sgm=error+mu
                opt_yx = func(x, 0, sgm)*Nx
                opt_yy = func(x, 0, sgm)*Ny
                ax[0].plot(x, opt_yx, c='b',label=(r'optimum: $\mu$='+str(round(mu,2))+', $\sigma$='+str(round(sgm,2))+'nm'))
                ax[1].plot(x, opt_yy, c='b',label=(r'optimum: $\mu$='+str(round(mu,2))+', $\sigma$='+str(round(sgm,2))+'nm'))
                if np.max([np.max(opt_yx),np.max(opt_yy)])>ymax: ymax=np.max([np.max(opt_yx),np.max(opt_yy)])*1.1
        else: ymax=np.max([np.max(nx[0]),np.max(ny[0])])*1.1


        ax[0].set_ylim([0,ymax])
        ax[0].set_xlim(-xlim,xlim)
        ax[0].set_xlabel('x-error [nm]')
        ax[0].set_ylabel('# of localizations')
        ax[0].legend(loc='upper right')
        
        ax[1].set_ylim([0,ymax])
        ax[1].set_xlim(-xlim,xlim)
        ax[1].set_xlabel('y-error [nm]')
        ax[1].set_ylabel('# of localizations')
        ax[1].legend(loc='upper right')
        fig.tight_layout()
        if annotate is not None: annotate_image(ax[0], annotate, displacement=[-40,2])
    

##fig1g
ErrorDistribution_xy(DS2, nbins=nbins, xlim=pair_filter[2], error=DS2.coloc_error, annotate='E')

##fig1h
fig,axFOV=DS2.ErrorFOV(None, figsize=(9,12), placement='bottom')
annotate_image(axFOV[0], 'F', displacement=[-10,2])