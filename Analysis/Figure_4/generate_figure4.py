# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:36:49 2021

@author: Mels
"""
    
from matplotlib import pyplot as plt
import tensorflow as tf
import copy
import numpy as np
from scipy.optimize import curve_fit
import scipy.special as scpspc

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')
output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_4/Figures/'

from dataset import dataset
from Channel import Channel
plt.rc('font', size=10)
dpi=800
    

#%% Load HEL1.hdf5
maxDistance=1000
DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
               'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
              pix_size=159, loc_error=10, mu=0, coloc_error=np.sqrt(2)*(10),
              imgshape=[512,256], linked=False, FrameLinking=True, BatchOptimization=False)
DS1.load_dataset_hdf5(align_rcc=True, transpose=True)

## optimization params
execute_linked=True
learning_rate=2e-3
epochs=300
pair_filter = [250, 180, 180]
gridsize=84
nbins=100
    

plt.close('all')
#%% fig 1
def plot_clusters(DS1, fig, ax):
    temp=copy.deepcopy(DS1)
    ax[0].plot(temp.ch1.ClusterCOM()[0][:,0],temp.ch1.ClusterCOM()[0][:,1], '.')
    ax[1].plot(temp.ch2.ClusterCOM()[0][:,0],temp.ch2.ClusterCOM()[0][:,1], '.')
    return fig, ax

def plot_frame(DS1, fig, ax):
    temp=copy.deepcopy(DS1)
    temp.generate_channel(precision=temp.pix_size)
    channel1=np.flipud(temp.channel1)
    channel2=np.flipud(temp.channel2)
    
    # plotting all channels
    ax[0].imshow(channel1, extent=temp.axis)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    center=1000
    ax[0].text(x=ax[0].get_xlim()[1]-center,y=ax[0].get_ylim()[1]-center, s='ch1', color='red',ha='right', va='top',
               bbox=dict(boxstyle="square", ec='red', fc=(1., 0.8, 0.8), ))
    ax[0].add_patch(plt.Rectangle((-18000, -38500),
                                10000, 500, ec='white', fc='white', zorder=100))
    ax[0].text(-18500, -38500, r'10$\mu$m', ha='left', va='bottom', color='white')
    
    ax[1].imshow(channel2, extent=temp.axis)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].text(x=ax[1].get_xlim()[1]-center,y=ax[1].get_ylim()[1]-center, s='ch2', color='blue',ha='right', va='top', 
               bbox=dict(boxstyle="square", ec='blue', fc=(0.8, 0.8, 1), ))    
    fig.tight_layout()
    return fig,ax
    

fig1=plt.figure(figsize=(1.95*2,1.95*11/6)) 
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)
ax=[ax1,ax2]
fig1,ax = plot_frame(DS1, fig1, ax) ## fig 1a

window=np.array([[1000, 21000],[-10000, 10000]])
ax[0].add_patch(plt.Rectangle((window[1,0],window[0,0]), window[1,1]-window[1,0], window[0,1]-window[0,0], 
                                ec='yellow', fc='none', linewidth=1, zorder=10))
ax[1].add_patch(plt.Rectangle((window[1,0],window[0,0]), window[1,1]-window[1,0], window[0,1]-window[0,0], 
                                ec='yellow', fc='none', linewidth=1, zorder=10))
fig1.savefig(output_path+'fig1', transparent=True, dpi=dpi) 


#%%  zoom1
def plot_zoom(DS1, window):
    temp=copy.deepcopy(DS1)
    temp=temp.SubsetWindow(window=window)
    fig=plt.figure(figsize=(1.95*2,1.95))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax=[ax1,ax2]
    
    temp.generate_channel(precision=temp.pix_size/5)
    #temp.axis = np.concatenate([ temp.bounds[1,:], temp.bounds[0,:]], axis=0) * temp.pix_size/5
    channel1=np.flipud(temp.channel1)
    channel2=np.flipud(temp.channel2)
    
    # plotting all channels
    ax[0].imshow(channel1, extent=temp.axis)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    center=500
    ax[0].text(x=ax[0].get_xlim()[1]-center,y=ax[0].get_ylim()[1]-center, s='ch1', color='red',ha='right', va='top',
               bbox=dict(boxstyle="square", ec='red', fc=(1., 0.8, 0.8), ))
    ax[0].add_patch(plt.Rectangle((window[1,0]+1000, window[0,0]+1000),
                                5000, 250, ec='white', fc='white', zorder=100))
    ax[0].text(window[1,0]+1000, window[0,0]+1000, r'5$\mu$m', ha='left', va='bottom', color='white')
    
    ax[1].imshow(channel2, extent=temp.axis)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].text(x=ax[1].get_xlim()[1]-center,y=ax[1].get_ylim()[1]-center, s='ch2', color='blue',ha='right', va='top', 
               bbox=dict(boxstyle="square", ec='blue', fc=(0.8, 0.8, 1), ))    
    fig.tight_layout()
    return fig,ax 

fig01, ax01=plot_zoom(DS1, window)
fig01.savefig(output_path+'fig2', transparent=True, dpi=dpi)


#%%
def saveDS(DS1, name, output_path):
    np.savetxt(output_path+name+'ch1_pos.txt',DS1.ch1.pos.numpy())
    np.savetxt(output_path+name+'ch2_pos.txt',DS1.ch2.pos.numpy())
    np.savetxt(output_path+name+'ch1_frame.txt',DS1.ch1.frame.numpy())
    np.savetxt(output_path+name+'ch2_frame.txt',DS1.ch2.frame.numpy())
    
def openDS(DS1, name, output_path):
    del DS1.ch1, DS1.ch2
    ch1_pos=np.loadtxt(output_path+name+'ch1_pos.txt')
    ch2_pos=np.loadtxt(output_path+name+'ch2_pos.txt')
    ch1_frame=np.loadtxt(output_path+name+'ch1_frame.txt')
    ch2_frame=np.loadtxt(output_path+name+'ch2_frame.txt')
    DS1.ch1=Channel(ch1_pos, ch1_frame)
    DS1.ch2=Channel(ch2_pos, ch2_frame)
    return DS1
    
    
if True:
    DS1,DS2=DS1.SplitDatasetClusters()
    AffineClusters=DS1.AffineLLS_clusters()
    DS1.Apply_Affine(AffineClusters)
    DS2.Apply_Affine(AffineClusters)
    
    fig01, ax01=plot_zoom(DS1, window)

    #%
    DS1.link_dataset(maxDistance=maxDistance)
    DS1.AffineLLS()
    DS1.Filter(pair_filter[0]) 
    
    DS2.Apply_Affine(DS1.AffineMat)
    DS2.link_dataset(maxDistance=maxDistance)
    #DS2.Filter(pair_filter[0]) 
       
    #% running the model
    DS1.AffineLLS()
    DS1.Apply_Affine(AffineClusters)
    DS1.Filter(pair_filter[0]) 
    
    saveDS(DS1, 'DS1', output_path)
    saveDS(DS2, 'DS2', output_path)
    
else:
    DS1=openDS(DS1, 'DS1', output_path)
    DS2=copy.deepcopy(DS1)
    DS2=openDS(DS2, 'DS2', output_path)
    DS1.linked=True
    DS2.linked=True

#% CatmullRomSplines
if epochs is not None:
    DS1.execute_linked=True #%% Splines can only be optimized by pair-optimization
    DS1.Train_Splines(learning_rate, epochs, gridsize, edge_grids=1)
    DS1.Apply_Splines()
DS1.Filter(pair_filter[1])

#% apply transformation to DS2
DS2.copy_models(DS1) ## Copy all mapping parameters
DS2.Apply_Splines()
DS2.Filter(pair_filter[1])
    


    

#%% fig 2
def plt_grid(DS1, fig=None,ax=None, locs_markersize=25, d_grid=.1, Ngrids=1, window0=None):
    print('Plotting...')
    def gridmapping(ax, DS1, d_grid, Ngrids, DSoriginal, lw=.1):
        ## Horizontal Grid
        Hx1_grid = tf.range(DS1.x1_min, DS1.x1_max, delta=d_grid, dtype=tf.float32)*DS1.gridsize
        Hx2_grid = tf.range(DS1.x2_min, DS1.x2_max, delta=1/Ngrids, dtype=tf.float32)*DS1.gridsize
        HGrid = tf.Variable( tf.reshape(tf.stack(tf.meshgrid(Hx1_grid, Hx2_grid), axis=-1) , (-1,2)) , trainable=False, dtype=tf.float32)
        ## Vertical Grid
        Vx1_grid = tf.range(DS1.x1_min, DS1.x1_max, delta=1/Ngrids, dtype=tf.float32)*DS1.gridsize
        Vx2_grid = tf.range(DS1.x2_min, DS1.x2_max, delta=d_grid, dtype=tf.float32)*DS1.gridsize
        VGrid = tf.Variable(tf.gather(tf.reshape(tf.stack(tf.meshgrid(Vx2_grid, Vx1_grid), axis=-1) , (-1,2)), [1,0], axis=1), trainable=False, dtype=tf.float32)
        # map the grids
        HGrid = DSoriginal.InputSplines(DSoriginal.SplinesModel( DSoriginal.InputSplines(HGrid) ), inverse=True)
        VGrid = DSoriginal.InputSplines(DSoriginal.SplinesModel( DSoriginal.InputSplines(VGrid) ), inverse=True)
        #HGrid = DS1.SplinesModel( DS1.InputSplines(HGrid) )*DS1.gridsize
        #VGrid = DS1.SplinesModel( DS1.InputSplines(VGrid) )*DS1.gridsize
                
        (nn, i,j)=(Hx1_grid.shape[0],0,0)
        while i<HGrid.shape[0]:
            if j%Ngrids==0:
                ax.plot(HGrid[i:i+nn,1], HGrid[i:i+nn,0], c='c', lw=lw)
            else:
                ax.plot(HGrid[i:i+nn,1], HGrid[i:i+nn,0], c='b', lw=lw)
            i+=nn
            j+=1
        (nn, i,j)=(Vx2_grid.shape[0],0,0)
        while i<VGrid.shape[0]:
            if j%Ngrids==0:
                ax.plot(VGrid[i:i+nn,1], VGrid[i:i+nn,0], c='c', lw=lw)
            else:
                ax.plot(VGrid[i:i+nn,1], VGrid[i:i+nn,0], c='b', lw=lw)
            i+=nn
            j+=1
            
    #'''
    ## Main figure
    #fig0=plt.figure(figsize=(1.95/2,1.95)) 
    fig0=plt.figure(figsize=(6,3)) 
    ax0 = fig0.add_subplot(111)  
    temp=copy.deepcopy(DS1)
    temp.generate_channel(precision=temp.pix_size)
    channel1=np.flipud(temp.channel1)
    ax0.imshow(channel1, extent=temp.axis)
    ax0.set_yticks([])
    ax0.set_xticks([])
    
    ## add measure
    ax0.add_patch(plt.Rectangle((-16000, -37500),
                                10000, 500, ec='white', fc='white', zorder=100))
    ax0.text(-16000, -37500, r'10$\mu$m', ha='left', va='bottom', color='white')
    
    #'''
    
    ## create zoom 
    temp0=copy.deepcopy(DS1)
    if window0 is None: window0=np.array([[-10000, 10000],[-15000, 15000]])
    temp0=temp0.SubsetWindow(window=window0)
    
    ## plotting zoom
    fig1=plt.figure(figsize=(1.95,1.95)) 
    ax1 = fig1.add_subplot(111)      
    temp0.generate_channel(precision=temp0.pix_size/5, bounds=window0)
    channel1=np.flipud(temp0.channel1)
    ax1.imshow(channel1, extent=temp0.axis)
    xlim=ax1.get_xbound()
    ylim=ax1.get_ybound()
    gridmapping(ax1, temp0, d_grid, Ngrids=1, DSoriginal=DS1)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax0.add_patch(plt.Rectangle((xlim[0],ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                                ec='red', fc='none', linewidth=1, zorder=100))
    
    ## create measure
    ax1.add_patch(plt.Rectangle((-7000, 2500),
                                5000, 250, ec='white', fc='white', zorder=100))
    ax1.text(-7000, 2500, r'5$\mu$m', ha='left', va='bottom', color='white')
    
    ## second zoom
    temp1=copy.deepcopy(DS1)
    windowx=650
    windowy=15000
    window1=np.array([[windowy, windowy+5000],[windowx, windowx+5000]])
    temp1=temp1.SubsetWindow(window=window1)
    
    ## plot second measure
    fig2=plt.figure(figsize=(1.95,1.95))
    ax2=fig2.add_subplot(111)
    temp1.generate_channel(precision=temp1.pix_size/5, bounds=window1)
    channel1=np.flipud(temp1.channel1)
    ax2.imshow(channel1, extent=temp1.axis)
    xlim=ax2.get_xbound()
    ylim=ax2.get_ybound()
    gridmapping(ax2, DS1, d_grid/Ngrids, Ngrids=Ngrids, DSoriginal=DS1, lw=.5)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax1.add_patch(plt.Rectangle((xlim[0],ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                                ec='red', fc='none', linewidth=1, zorder=100))
    
    ## add measure
    ax2.add_patch(plt.Rectangle((-12800, -5750),
                                1000, 50, ec='white', fc='white', zorder=100))
    ax2.text(-12800, -5750, r'1$\mu$m', ha='left', va='bottom', color='white')
    
    def create_zoom(DS1, window2):
        temp2=copy.deepcopy(DS1)
        temp2=temp2.SubsetWindow(window=window2)
        
        ## plot second measure
        fig3=plt.figure(figsize=(1.95*3/4,1.95*3/4))
        ax3=fig3.add_subplot(111)
        temp2.generate_channel(precision=temp2.pix_size/5, bounds=window2)
        channel1=np.flipud(temp2.channel1)
        ax3.imshow(channel1, extent=temp2.axis)
        xlim=ax3.get_xbound()
        ylim=ax3.get_ybound()
        gridmapping(ax3, DS1, d_grid/Ngrids, Ngrids=Ngrids, DSoriginal=DS1, lw=1)
        ax3.set_yticks([])
        ax3.set_xticks([])
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        return fig3, ax3
    
    windowx=720
    windowy=16400
    window2=np.array([[windowy, windowy+500],[windowx, windowx+500]])
    fig30,ax30=create_zoom(DS1, window2)
    ax2.add_patch(plt.Rectangle((windowx, windowy), 500, 500,ec='red', fc='none', linewidth=1, zorder=100))
    
    windowx=3610
    windowy=16600
    window2=np.array([[windowy, windowy+500],[windowx, windowx+500]])
    fig31,ax31=create_zoom(DS1, window2)
    ax2.add_patch(plt.Rectangle((windowx, windowy), 500, 500,ec='red', fc='none', linewidth=1, zorder=100))
    
    windowx=4650
    windowy=19200
    window2=np.array([[windowy, windowy+500],[windowx, windowx+500]])
    fig32,ax32=create_zoom(DS1, window2)
    ax2.add_patch(plt.Rectangle((windowx, windowy), 500, 500,ec='red', fc='none', linewidth=1, zorder=100))
    
    windowx=1480
    windowy=16300
    window2=np.array([[windowy, windowy+500],[windowx, windowx+500]])
    fig33,ax33=create_zoom(DS1, window2)
    ax2.add_patch(plt.Rectangle((windowx, windowy), 500, 500,ec='red', fc='none', linewidth=1, zorder=100))
    return fig1, ax1, fig2, ax2, [fig30, fig31, fig32, fig33], [ax30, ax31, ax32, ax33]
    
    
fig2, ax2, fig3, ax3, fig3zoom, ax3zoom=plt_grid(DS1, locs_markersize=25, d_grid=.1, Ngrids=1, window0=window) ## fig 1c
#fig20.savefig(output_path+'fig20', transparent=True, dpi=dpi) 
fig2.savefig(output_path+'fig3', transparent=True, dpi=dpi) 
fig3.savefig(output_path+'fig4', transparent=True, dpi=dpi) 
fig3zoom[0].savefig(output_path+'fig4zoom', transparent=True, dpi=dpi)
fig3zoom[1].savefig(output_path+'fig4zoom1', transparent=True, dpi=dpi)
fig3zoom[2].savefig(output_path+'fig4zoom2', transparent=True, dpi=dpi)
fig3zoom[3].savefig(output_path+'fig4zoom3', transparent=True, dpi=dpi) 



#%% fig 3
def ErrorDistribution_r(DS1, fig=None,ax=None, nbins=30, xlim=31, error=None, mu=None, fit_data=True):
    if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    if mu is None: mu=0
    pos1=DS1.ch1.pos_all()
    pos2=DS1.ch2.pos_all()
        
    # Calculating the error
    dist, avg, r = DS1.ErrorDist(pos1, pos2)
    dist=dist[np.argwhere(dist<xlim)]
    
    # plotting the histogram
    if fig is None: fig=plt.figure(figsize=(1.95*2,1.95*2/3)) 
    if ax is None: ax = fig.add_subplot(111)
    
    n=ax.hist(dist, range=[0,xlim], label='N='+str(pos1.shape[0]), alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
    ymax = np.max(n[0])*1.1
    
    
    if fit_data: ## fit bar plot data using curve_fit
        def func(r, sigma, mu=mu): # from Churchman et al 2006
            sigma2=sigma**2
            if mu==0: return r/sigma2*np.exp(-r**2/2/sigma2)
            else: return (r/sigma2)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)

        N = pos1.shape[0] * ( n[1][1]-n[1][0] )
        xn=(n[1][:-1]+n[1][1:])/2 
        popt, pcov = curve_fit(func, xn, n[0]/N, p0=np.std(xn))
        x = np.linspace(0, xlim, 1000)
        y = func(x, *popt)*N
        ax.plot(x, y, c='g',label=(r'$\sigma$='+str(round(popt[0],2))+'nm'))
    
        ## plot how function should look like
        if error is not None:
            sgm=error
            y = func(x, sgm, mu)*N
            ax.plot(x, y, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm'))
            if np.max(y)>ymax: ymax=np.max(y)*1.1

    # Some extra plotting parameters
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xlim])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

fig4=plt.figure(figsize=(1.95*2,1.95))
ax1 = fig4.add_subplot(211)
ax2 = fig4.add_subplot(212)

fig4, ax1=ErrorDistribution_r(DS1, fig4, ax1, nbins=nbins, xlim=pair_filter[2], 
                             error=DS1.coloc_error, mu=DS1.mu, fit_data=True)
fig4, ax2=ErrorDistribution_r(DS2, fig4, ax2, nbins=nbins, xlim=pair_filter[2],
                             error=DS2.coloc_error, mu=DS2.mu, fit_data=True)

ylim=np.max([ax1.get_ylim()[1],ax2.get_ylim()[1]])
ax1.set_ylim([0,ylim])
ax2.set_ylim([0,ylim])
ax2.set_xticks([0,int(pair_filter[2]/3), int(pair_filter[2]*2/3),pair_filter[2]])
ax2.set_xlabel('absolute error [nm]')
fig4.tight_layout(h_pad=0)
ax1.text(x=pair_filter[2]*.97,y=ylim*.55, s=r'Training', color='black',ha='right', va='top')
ax2.text(x=pair_filter[2]*.97,y=ylim*.55, s=r'Cross-Validation', color='black',ha='right', va='top')
ax1.text(x=pair_filter[2]*.97,y=ylim*.3, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax2.text(x=pair_filter[2]*.97,y=ylim*.3, s=ax2.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig4.savefig(output_path+'fig5', transparent=True, dpi=dpi) 

txt='CRO Training: '+ax1.get_legend_handles_labels()[1][0]+'\nCRO Cross-Validation: '+ax2.get_legend_handles_labels()[1][0]+'\nLower Bound: '+ax1.get_legend_handles_labels()[1][1]
file2write=open(output_path+'fig5.txt','w')
file2write.write(txt)
file2write.close()


#%% fig1ef
def ErrorDistribution_xy(DS1, nbins=30, xlim=31, error=None, mu=None, fit_data=True):
        if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        if mu is None: mu=0
        pos1=DS1.ch1.pos_all()
        pos2=DS1.ch2.pos_all()
            
        fig, ax = plt.subplots(1,2,figsize=(1.95*4,1.95))
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
            ax[0].plot(x, yx, c='g',label=(r'$\sigma$='+str(round(poptx[1],2))+'nm\n$\mu$='+str(round(poptx[0],2))+'nm'))
            ax[1].plot(x, yy, c='g',label=(r'$\sigma$='+str(round(popty[1],2))+'nm\n$\mu$='+str(round(popty[0],2))+'nm'))
            ymax = np.max([np.max(nx[0]),np.max(ny[0]), np.max(yx), np.max(yy)])*1.1
            
            ## plot how function should look like
            if error is not None:
                sgm=error+mu
                opt_yx = func(x, 0, sgm)*Nx
                opt_yy = func(x, 0, sgm)*Ny
                ax[0].plot(x, opt_yx, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm'))
                ax[1].plot(x, opt_yy, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm'))
                if np.max([np.max(opt_yx),np.max(opt_yy)])>ymax: ymax=np.max([np.max(opt_yx),np.max(opt_yy)])*1.1
        else: ymax=np.max([np.max(nx[0]),np.max(ny[0])])*1.1


        ax[0].set_ylim([0,ymax])
        ax[0].set_xlim(-xlim,xlim)
        #ax[0].text(x=-xlim+1,y=ymax, s='x-error', color='black',ha='left', va='top')
        ax[0].set_xlabel('x-error [nm]')
        ax[0].set_xticks([-int(xlim*2/3), 0, int(xlim*2/3)])
        ax[0].set_yticks([])
        
        ax[1].set_ylim([0,ymax])
        ax[1].set_xlim(-xlim,xlim)
        #ax[1].text(x=-xlim+1,y=ymax, s='y-error', color='black',ha='left', va='top')
        ax[1].set_xlabel('y-error [nm]')
        ax[1].set_xticks([-int(xlim*2/3), 0, int(xlim*2/3)])
        ax[1].set_yticks([])
        
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['left'].set_visible(False) 
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_visible(False) 
        return fig, ax
    

##fig1g
fig5,ax5=ErrorDistribution_xy(DS2, nbins=nbins, xlim=pair_filter[2], error=DS2.coloc_error)
ax5[0].text(x=pair_filter[2]*.7,y=ylim*.9, s=ax5[0].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax5[1].text(x=pair_filter[2]*.7,y=ylim*.9, s=ax5[1].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig5.tight_layout()
fig5.savefig(output_path+'fig6', transparent=True, dpi=dpi) 

txt='CRO Training: '+ax5[0].get_legend_handles_labels()[1][0]+'\nCRO Cross-Validation: '+ax5[1].get_legend_handles_labels()[1][0]+'\nLower Bound: '+ax5[0].get_legend_handles_labels()[1][1]
file2write=open(output_path+'fig6.txt','w')
file2write.write(txt)
file2write.close()
'''
##fig1h
fig6,ax6=DS2.ErrorFOV(None, figsize=(1.95*3/2,1.95*2))
fig6.savefig(output_path+'fig7', transparent=True, dpi=dpi) 
'''
