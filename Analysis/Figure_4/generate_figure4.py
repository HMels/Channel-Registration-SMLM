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
from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')
output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_4/Figures/'

from dataset import dataset
from Channel import Channel
plt.rc('font', size=10)
dpi=800
    

#%% Load HEL1.hdf5
nbins=100
maxDistance=300
k=8
opt_fn=tf.optimizers.SGD
DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
               'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
              pix_size=159, loc_error=10, mu=0, imgshape=[512,256], 
              linked=False, FrameLinking=True, BatchOptimization=False, execute_linked=False)
DS1.load_dataset_hdf5(align_rcc=True, transpose=True)
    

plt.close('all')
#%% fig 1
def plot_2channels(DS1, figsize):    
    temp=copy.deepcopy(DS1)
    temp.pix_size=100
    fig, ax=temp.show_channel(temp.ch1.pos, color='blue', figsize=figsize, ps=1, alpha=1, addpatch=False)
    fig, ax=temp.show_channel(temp.ch2.pos, color='red', fig=fig, ax=ax, ps=1, alpha=.7,  addpatch=True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.tight_layout()  
    return fig, ax
    

fig1, ax1=plot_2channels(DS1, figsize=(3.9,1.95))
window=np.array([[-18500,-3500],[-7500, 7500]])
ax1.add_patch(plt.Rectangle((window[0,0]/1000,window[1,0]/1000), (window[0,1]-window[0,0])/1000,
                            (window[1,1]-window[1,0])/1000, ec='red', fc='none', linewidth=1, zorder=10))
fig1.savefig(output_path+'fig1', transparent=True, dpi=dpi) 


#%%  zoom1
def plot_zoom(DS1, window, figsize):
    temp=copy.deepcopy(DS1)
    temp=temp.SubsetWindow(window=window)
    
    # plotting all channels
    fig, ax=temp.show_channel(temp.ch1.pos, color='blue', figsize=figsize, 
                              ps=1, alpha=1, addpatch=False, lims=window)
    fig, ax=temp.show_channel(temp.ch2.pos, color='red', fig=fig, ax=ax,
                              ps=1, alpha=.7,  addpatch=False, lims=window)
    x1=np.min(temp.ch1.pos[:,0])/1000 +1
    x2=np.min(temp.ch1.pos[:,1])/1000 +1
    ax.add_patch(Rectangle((x1,x2), 5, .5, ec='black', fc='black'))
    ax.text(x1, x2+.5, r'5$\mu$m', ha='left', va='bottom')
    return fig,ax 

fig2, ax2=plot_zoom(DS1, window, figsize=(1.53,1.53))
fig2.savefig(output_path+'fig2', transparent=True, dpi=dpi)

#%% AffineLLS cluster basis 
DS01=copy.deepcopy(DS1)
DS01clust=DS01.ClusterDataset(loc_error=None)
DS01clust.execute_linked=True
DS01clust.link_dataset(maxDistance=maxDistance)

## optimization params
learning_rate=5e-5
epochs=1000
pair_filter=[None, None, 180]
gridsize=7500

#% aligning clusters
DS01clust.AffineLLS(maxDistance, k)
DS01.copy_models(DS01clust) ## Copy all mapping parameters
DS01.Apply_Affine(DS01clust.AffineMat)
DS01.Filter(pair_filter[0]) 

fig3, ax3=plot_zoom(DS01, window, figsize=(1.95,1.95))
fig3.savefig(output_path+'fig3', transparent=True, dpi=dpi)


#% CatmullRomSplines
if epochs is not None:
    DS01.execute_linked=True
    if not DS01.linked: DS01.link_dataset(maxDistance=maxDistance)
    DS01.Train_Splines(learning_rate, 300, gridsize, edge_grids=1, opt_fn=opt_fn, 
                      maxDistance=maxDistance, k=k)
    DS01.Apply_Splines()
    

#%% Spline grid plotting
def plt_grid(DS1, fig=None,ax=None, figsize=(1.95,1.95),
             locs_markersize=25, d_grid=.1, Ngrids=1, window=None):
    print('Plotting...')
    ## Main figure
    temp=copy.deepcopy(DS1)
    temp=temp.SubsetWindow(window=window)
    offset=temp.zero_image(2*temp.gridsize)
    temp.gridsize/=1000
    fig0, ax0=temp.show_channel(temp.ch1.pos, color='blue', figsize=figsize,
                                ps=1, alpha=1, addpatch=False, lims=window+offset[:,None])
    fig0, ax0=temp.show_channel(temp.ch2.pos, color='red', fig=fig0, ax=ax0,
                                ps=1, alpha=.7,  addpatch=False, lims=window+offset[:,None])
    #fig0, ax0=temp.PlotSplineGrid(fig=fig0, ax=ax0)
    #gridmapping(ax0, temp, d_grid, Ngrids=1, DSoriginal=DS1, lw=10)
    x1=np.min(temp.ch1.pos[:,0])/1000 +1
    x2=np.min(temp.ch1.pos[:,1])/1000 +1
    ax0.add_patch(Rectangle((x1,x2), 5, .5, ec='black', fc='black'))
    ax0.text(x1, x2+.5, r'5$\mu$m', ha='left', va='bottom')   
    
    fig00, ax00=temp.show_channel(temp.ch1.pos, color='blue', figsize=figsize,
                                ps=1, alpha=1, addpatch=False, lims=window+offset[:,None])
    fig00, ax00=temp.show_channel(temp.ch2.pos, color='red', fig=fig00, ax=ax00,
                                ps=1, alpha=.7,  addpatch=False, lims=window+offset[:,None])
    
    x1=19139
    x2=28730
    window0=np.array([[x1, x1+300],[x2, x2+300]])
    temp0=copy.deepcopy(temp)
    temp0=temp0.SubsetWindow(window=window0)
    fig30, ax30=temp.show_channel(temp0.ch1.pos, color='blue', figsize=(1.14,1.14),
                                ps=3, alpha=1, addpatch=False, lims=window0)
    fig30, ax30=temp.show_channel(temp0.ch2.pos, color='red', fig=fig30, ax=ax30,
                                ps=3, alpha=1,  addpatch=False, lims=window0)
    #ax0.add_patch(plt.Rectangle((window0[0,0]/1000,window0[1,0]/1000), (window0[0,1]-window0[0,0])/1000,
      #                      (window0[1,1]-window0[1,0])/1000, ec='green', fc='none', linewidth=1.5, zorder=10))
    ax00.annotate('1', (x1/1000,x2/1000), color='red')
    
    
    x1=18235
    x2=21640
    window0=np.array([[x1, x1+300],[x2, x2+300]])
    temp0=copy.deepcopy(temp)
    temp0=temp0.SubsetWindow(window=window0)
    fig31, ax31=temp.show_channel(temp0.ch1.pos, color='blue', figsize=(1.14,1.14),
                                ps=3, alpha=1, addpatch=False, lims=window0)
    fig31, ax31=temp.show_channel(temp0.ch2.pos, color='red', fig=fig31, ax=ax31,
                                ps=3, alpha=1,  addpatch=False, lims=window0)
    #ax0.add_patch(plt.Rectangle((window0[0,0]/1000,window0[1,0]/1000), (window0[0,1]-window0[0,0])/1000,
     #                       (window0[1,1]-window0[1,0])/1000, ec='blue', fc='none', linewidth=1.5, zorder=10))
    ax00.annotate('2', (x1/1000,x2/1000), color='red')
    
    
    x1=26331
    x2=25500
    window0=np.array([[x1, x1+300],[x2, x2+300]])
    temp0=copy.deepcopy(temp)
    temp0=temp0.SubsetWindow(window=window0)
    fig32, ax32=temp.show_channel(temp0.ch1.pos, color='blue', figsize=(1.14,1.14),
                                ps=3, alpha=1, addpatch=False, lims=window0)
    fig32, ax32=temp.show_channel(temp0.ch2.pos, color='red', fig=fig32, ax=ax32,
                                ps=3, alpha=1,  addpatch=False, lims=window0)
    #ax0.add_patch(plt.Rectangle((window0[0,0]/1000,window0[1,0]/1000), (window0[0,1]-window0[0,0])/1000,
     #                       (window0[1,1]-window0[1,0])/1000, ec='purple', fc='none', linewidth=1.5, zorder=10))
    ax00.annotate('3', (x1/1000,x2/1000), color='red')
    
    x1=22800
    x2=16930
    window0=np.array([[x1, x1+300],[x2, x2+300]])
    temp0=copy.deepcopy(temp)
    temp0=temp0.SubsetWindow(window=window0)
    fig33, ax33=temp.show_channel(temp0.ch1.pos, color='blue', figsize=(1.14,1.14),
                                ps=3, alpha=1, addpatch=False, lims=window0)
    fig33, ax33=temp.show_channel(temp0.ch2.pos, color='red', fig=fig33, ax=ax33,
                                ps=3, alpha=1,  addpatch=False, lims=window0)
    #ax0.add_patch(plt.Rectangle((window0[0,0]/1000,window0[1,0]/1000), (window0[0,1]-window0[0,0])/1000,
     #                       (window0[1,1]-window0[1,0])/1000, ec='purple', fc='none', linewidth=1.5, zorder=10))
    ax00.annotate('4', (x1/1000,x2/1000), color='red')
    
    
    return fig0, ax0,fig00, ax00, [fig30, fig31, fig32, fig33], [ax30, ax31, ax32, ax33]
    
    
fig4, ax4,fig40, ax40, fig4zoom, ax4zoom=plt_grid(DS01, figsize=(1.95,1.95), locs_markersize=25, d_grid=.1, Ngrids=1, window=window) ## fig 1c
fig4.savefig(output_path+'fig4', transparent=True, dpi=dpi) 
fig40.savefig(output_path+'fig4annotated', transparent=True, dpi=dpi) 

fig4zoom[0].savefig(output_path+'fig4zoom', transparent=True, dpi=dpi)
fig4zoom[1].savefig(output_path+'fig4zoom1', transparent=True, dpi=dpi)
fig4zoom[2].savefig(output_path+'fig4zoom2', transparent=True, dpi=dpi)
fig4zoom[3].savefig(output_path+'fig4zoom3', transparent=True, dpi=dpi)

'''
#%% complete dataset
DS02=copy.deepcopy(DS1)
DS02.copy_models(DS01clust) ## Copy all mapping parameters
DS02.Apply_Affine(DS01clust.AffineMat)
if DS02.SplinesModel is not None: DS02.Apply_Splines()
DS01.Filter(pair_filter[1])
DS02.Apply_Affine(DS01.AffineMat)
DS02.copy_models(DS01) ## Copy all mapping parameters
if DS02.SplinesModel is not None: DS02.Apply_Splines()
fig5, ax5=plot_zoom(DS02, window, figsize=(1.95,1.95))
fig5.savefig(output_path+'fig5', transparent=True, dpi=dpi) 
'''
#%% AffineLLS cluster split 
DS1, DS2=DS1.SplitDatasetClusters()
DS1clust=DS1.ClusterDataset(loc_error=None)
DS1clust.execute_linked=True
DS2clust=DS2.ClusterDataset(loc_error=None)
DS1clust.link_dataset(maxDistance=maxDistance)

## optimization params
learning_rate=5e-5
epochs=1000
pair_filter=[None, None, maxDistance]
gridsize=7500

#% aligning clusters
DS1clust.AffineLLS(maxDistance, k)
DS1.copy_models(DS1clust) ## Copy all mapping parameters
DS1.Apply_Affine(DS1clust.AffineMat)
if DS2clust is not None:
    DS2.copy_models(DS1clust) ## Copy all mapping parameters
    DS2.Apply_Affine(DS1clust.AffineMat)
DS1.Filter(pair_filter[0]) 

#% CatmullRomSplines
if epochs is not None:
    DS1.execute_linked=True
    if not DS1.linked: DS1.link_dataset(maxDistance=maxDistance)
    DS1.Train_Splines(learning_rate, 300, gridsize, edge_grids=1, opt_fn=opt_fn, 
                      maxDistance=maxDistance, k=k)
    DS1.Apply_Splines()
    
DS1.Filter(pair_filter[1])

if DS2 is not None:
    DS2.copy_models(DS1) ## Copy all mapping parameters
    if DS2.SplinesModel is not None: DS2.Apply_Splines()
    if not DS2.linked: DS2.link_dataset(maxDistance=maxDistance)
    DS2.Filter(pair_filter[1])
   

#%% fig 3
pair_filter[2]=180
def ErrorDistribution_r(DS1, fig=None,ax=None, nbins=30, xlim=31, error=None, fit_data=True):
    if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    pos1=DS1.ch1.pos_all()
    pos2=DS1.ch2.pos_all()
        
    # Calculating the error
    dist, avg, r = DS1.ErrorDist(pos1, pos2)
    dist=dist[np.argwhere(dist<xlim)]
    
    # plotting the histogram
    if fig is None: fig=plt.figure(figsize=(2.5,1.95*2/3)) 
    if ax is None: ax = fig.add_subplot(111)
    
    n=ax.hist(dist, range=[0,xlim], label='N='+str(pos1.shape[0]), alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
    ymax = np.max(n[0])*1.1
    
    
    if fit_data: ## fit bar plot data using curve_fit
        def func(r, sigma, mu): # from Churchman et al 2006
            sigma2=sigma**2
            if mu==0: return r/sigma2*np.exp(-r**2/2/sigma2)
            else: return (r/sigma2)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)

        N = pos1.shape[0] * ( n[1][1]-n[1][0] )
        xn=(n[1][:-1]+n[1][1:])/2 
        popt, pcov = curve_fit(func, xn, n[0]/N, p0=[np.std(xn), np.average(xn)])
        x = np.linspace(0, xlim, 1000)
        y = func(x, *popt)*N
        ax.plot(x, y, c='g',label=(r'$\sigma$='+str(round(popt[0],2))+'nm\n$\mu$='+str(round(popt[1],2))+'nm'))
    
        ## plot how function should look like
        if error is not None:
            sgm=error
            y = func(x, sgm, 0)*N
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

fig6=plt.figure(figsize=(4.5,2.15))
ax61 = fig6.add_subplot(211)
ax62 = fig6.add_subplot(212)

fig6, ax1=ErrorDistribution_r(DS1, fig6, ax61, nbins=nbins, xlim=pair_filter[2], 
                             error=DS1.coloc_error, fit_data=True)
fig6, ax2=ErrorDistribution_r(DS2, fig6, ax62, nbins=nbins, xlim=pair_filter[2],
                             error=DS2.coloc_error, fit_data=True)

ylim=np.max([ax61.get_ylim()[1],ax62.get_ylim()[1]])
ax61.set_ylim([0,ylim])
ax62.set_ylim([0,ylim])
ax62.set_xticks([0,int(pair_filter[2]/3), int(pair_filter[2]*2/3),pair_filter[2]])
ax62.set_xlabel('absolute error [nm]')
fig6.tight_layout(h_pad=0)
ax61.text(x=pair_filter[2]*.97,y=ylim*.85, s='$CRsCR$ estimation', color='black',ha='right', va='top')
ax62.text(x=pair_filter[2]*.97,y=ylim*.85, s='$CRsCR$ testing', color='black',ha='right', va='top')
ax61.text(x=pair_filter[2]*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax62.text(x=pair_filter[2]*.97,y=ylim*.6, s=ax2.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig6.savefig(output_path+'fig6', transparent=True, dpi=dpi) 

txt='CRO Training: '+ax1.get_legend_handles_labels()[1][0]+'\nCRO Cross-Validation: '+ax2.get_legend_handles_labels()[1][0]+'\nLower Bound: '+ax1.get_legend_handles_labels()[1][1]
file2write=open(output_path+'fig6.txt','w')
file2write.write(txt)
file2write.close()


#%% fig1ef
pair_filter[2]=120
def ErrorDistribution_xy(DS1, DS2, nbins=30, xlim=31, error=None, mu=None, fit_data=True):
        if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        if mu is None: mu=0
        fig, ax = plt.subplots(4,1,figsize=(2*1.95,2*1.95))
        pos1=DS1.ch1.pos_all()
        pos2=DS1.ch2.pos_all()
            
        distx=pos1[:,0]-pos2[:,0]
        disty=pos1[:,1]-pos2[:,1]
        mask=np.where(distx<xlim,True, False)*np.where(disty<xlim,True,False)
        distx=distx[mask]
        disty=disty[mask]
        nx = ax[0].hist(distx, range=[-xlim,xlim], label='N='+str(pos1.shape[0]),alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        ny = ax[2].hist(disty, range=[-xlim,xlim], label='N='+str(pos1.shape[0]),alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        
        
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
            ax[2].plot(x, yy, c='g',label=(r'$\sigma$='+str(round(popty[1],2))+'nm\n$\mu$='+str(round(popty[0],2))+'nm'))
            ymax = np.max([np.max(nx[0]),np.max(ny[0]), np.max(yx), np.max(yy)])*1.1
            
            ## plot how function should look like
            if error is not None:
                sgm=error+mu
                opt_yx = func(x, 0, sgm)*Nx
                opt_yy = func(x, 0, sgm)*Ny
                ax[0].plot(x, opt_yx, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm'))
                ax[2].plot(x, opt_yy, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm'))
                if np.max([np.max(opt_yx),np.max(opt_yy)])>ymax: ymax=np.max([np.max(opt_yx),np.max(opt_yy)])*1.1
        else: ymax=np.max([np.max(nx[0]),np.max(ny[0])])*1.1


        ax[0].set_ylim([0,ymax])
        ax[0].set_xlim(-xlim,xlim)
        #ax[0].text(x=-xlim+1,y=ymax, s='x-error', color='black',ha='left', va='top')
        #ax[0].set_xlabel('x-error [nm]')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        
        ax[2].set_ylim([0,ymax])
        ax[2].set_xlim(-xlim,xlim)
        #ax[2].text(x=-xlim+1,y=ymax, s='y-error', color='black',ha='left', va='top')
        #ax[2].set_xlabel('y-error [nm]')
        ax[2].set_yticks([])
        
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['left'].set_visible(False) 
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['left'].set_visible(False) 
        ax[2].set_xticks([])
        
        pos1=DS2.ch1.pos_all()
        pos2=DS2.ch2.pos_all()
        distx=pos1[:,0]-pos2[:,0]
        disty=pos1[:,1]-pos2[:,1]
        mask=np.where(distx<xlim,True, False)*np.where(disty<xlim,True,False)
        distx=distx[mask]
        disty=disty[mask]
        nx = ax[1].hist(distx, range=[-xlim,xlim], label='N='+str(pos1.shape[0]),alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        ny = ax[3].hist(disty, range=[-xlim,xlim], label='N='+str(pos1.shape[0]),alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        
        
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
            ax[1].plot(x, yx, c='g',label=(r'$\sigma$='+str(round(poptx[1],2))+'nm\n$\mu$='+str(round(poptx[0],2))+'nm'))
            ax[3].plot(x, yy, c='g',label=(r'$\sigma$='+str(round(popty[1],2))+'nm\n$\mu$='+str(round(popty[0],2))+'nm'))
            ymax = np.max([np.max(nx[0]),np.max(ny[0]), np.max(yx), np.max(yy)])*1.1
            
            ## plot how function should look like
            if error is not None:
                sgm=error+mu
                opt_yx = func(x, 0, sgm)*Nx
                opt_yy = func(x, 0, sgm)*Ny
                ax[1].plot(x, opt_yx, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm'))
                ax[3].plot(x, opt_yy, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm'))
                if np.max([np.max(opt_yx),np.max(opt_yy)])>ymax: ymax=np.max([np.max(opt_yx),np.max(opt_yy)])*1.1
        else: ymax=np.max([np.max(nx[0]),np.max(ny[0])])*1.1


        ax[1].set_ylim([0,ymax])
        ax[1].set_xlim(-xlim,xlim)
        #ax[0].text(x=-xlim+1,y=ymax, s='x-error', color='black',ha='left', va='top')
        ax[1].set_xlabel('x-error [nm]')
        ax[1].set_yticks([])
        
        ax[3].set_ylim([0,ymax])
        ax[3].set_xlim(-xlim,xlim)
        #ax[1].text(x=-xlim+1,y=ymax, s='y-error', color='black',ha='left', va='top')
        ax[3].set_xlabel('y-error [nm]')
        ax[3].set_yticks([])
        
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_visible(False) 
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['left'].set_visible(False)
        ax[1].set_xticks([]) 
        ax[3].set_xticks([-100, -50, 0, 50,100])
        return fig, ax
    

##fig1g
fig7,ax7=ErrorDistribution_xy(DS1, DS2, nbins=nbins, xlim=pair_filter[2], error=DS2.coloc_error)
ax7[0].text(x=-pair_filter[2]*.9,y=ylim*.65, s=r'$CRsCR$'+'\nestimation', color='black',ha='left', va='top')
ax7[1].text(x=-pair_filter[2]*.9,y=ylim*.65, s=r'$CRsCR$'+'\ntesting', color='black',ha='left', va='top')
ax7[0].text(x=pair_filter[2]*.9,y=ylim*.65, s=ax7[0].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax7[1].text(x=pair_filter[2]*.9,y=ylim*.65, s=ax7[1].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax7[2].text(x=-pair_filter[2]*.9,y=ylim*.65, s=r'$CRsCR$'+'\nestimation', color='black',ha='left', va='top')
ax7[3].text(x=-pair_filter[2]*.9,y=ylim*.65, s=r'$CRsCR$'+'\ntesting', color='black',ha='left', va='top')
ax7[2].text(x=pair_filter[2]*.9,y=ylim*.65, s=ax7[2].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax7[3].text(x=pair_filter[2]*.9,y=ylim*.65, s=ax7[3].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig7.tight_layout()
fig7.savefig(output_path+'fig7', transparent=True, dpi=dpi) 

txt='CRO Training: '+ax7[0].get_legend_handles_labels()[1][0]+'\nCRO Cross-Validation: '+ax7[1].get_legend_handles_labels()[1][0]+'\nLower Bound: '+ax7[0].get_legend_handles_labels()[1][1]
file2write=open(output_path+'fig7.txt','w')
file2write.write(txt)
file2write.close()


#%%
def ErrorFOV(DS1, DS2, maxDistance=30, ps=1, cmap='seismic', figsize=None, title=None, precision=750,
                 placement='right', clusters=False, text=True, alpha=.8):
    import matplotlib as mpl    
    def prepdata(pos, z, precision=750):
        min1=np.min(pos[:,0])/1000
        min2=np.min(pos[:,1])/1000
        max1=np.max(pos[:,0])/1000
        max2=np.max(pos[:,1])/1000
        density=pos.shape[0]/((max1-min1)*(max2-min2))*precision/1000
        pos[:,0]-=min1
        pos[:,1]-=min2
        
        N1=int(np.max(pos[:,0])/precision)+1
        N2=int(np.max(pos[:,1])/precision)+1
        Z=np.zeros([N2,N1], dtype=float)
        N=np.zeros([N2,N1], dtype=float)
        for n in range(z.shape[0]): #calculate average displacement per cell
            i=np.floor(pos[n,1]/precision).astype('int')
            j=np.floor(pos[n,0]/precision).astype('int')
            Z[i,j]+=z[n]/density
            N[i,j]+=1
        for i in range(N2):
            for j in range(N1):
                if N[i,j]>1.: Z[i,j]/=N[i,j]
        X=np.linspace(min1,max1, N1)
        Y=np.linspace(min2,max2, N2)
        return X,Y,Z
    
    pos1=DS1.ch1.pos.numpy()
    pos2=DS1.ch2.pos.numpy()
    dist = pos1-pos2
    mask=np.where(np.sqrt(np.sum(dist**2,axis=1))<maxDistance,True, False)
    dist=dist[mask]
    pos1=pos1[mask]
    
    pos11=DS2.ch1.pos.numpy()
    pos21=DS2.ch2.pos.numpy()
    dist1 = pos11-pos21
    mask=np.where(np.sqrt(np.sum(dist1**2,axis=1))<maxDistance,True, False)
    dist1=dist1[mask]
    pos11=pos11[mask]
        
    if figsize is None: figsize=(14,6)
    fig, ax = plt.subplots(2,2, figsize=figsize,sharex = False,sharey=False,constrained_layout=True)
    xlim=(tf.reduce_min(pos1[:,0]/1000),tf.reduce_max(pos1[:,0]/1000))
    ylim=(tf.reduce_min(pos1[:,1]/1000),tf.reduce_max(pos1[:,1]/1000))
    
    X1,Y1,Z1=prepdata(pos1, dist[:,0], precision=precision)
    X2,Y2,Z2=prepdata(pos1, dist[:,1], precision=precision)
    X3,Y3,Z3=prepdata(pos11, dist1[:,0], precision=precision)
    X4,Y4,Z4=prepdata(pos11, dist1[:,1], precision=precision)
    
    vmin=np.min([np.min(Z1),np.min(Z2),np.min(Z3),np.min(Z4)])
    vmax=np.max([np.max(Z1),np.max(Z2),np.max(Z3),np.max(Z4)])
    if vmin<0: 
        vm=np.max([-vmin, vmax])
        norm=mpl.colors.Normalize(vmin=-vm, vmax=vm, clip=False)
    else:
        norm=mpl.colors.Normalize(vmin=0, vmax=vmax, clip=False)
        
    ax[0,0].contourf(X1,Y1,Z1, norm=norm, cmap=cmap, alpha=.8)
    ax[0,1].contourf(X2,Y2,Z2, norm=norm, cmap=cmap, alpha=.8)
    ax[1,0].contourf(X3,Y3,Z3, norm=norm, cmap=cmap, alpha=.8)
    ax[1,1].contourf(X4,Y4,Z4, norm=norm, cmap=cmap, alpha=.8) 
         
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_xlim(xlim)
    ax[0,0].set_ylim(ylim)
    ax[0,0].set_title('x-error')
    ax[0,0].set_aspect('equal', 'box')
    
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,1].set_xlim(xlim)
    ax[0,1].set_ylim(ylim)
    ax[0,1].set_title('y-error')
    ax[0,1].set_aspect('equal', 'box')
    
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_xlim(xlim)
    ax[1,0].set_ylim(ylim)
    ax[1,0].set_aspect('equal', 'box')
    
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,1].set_xlim(xlim)
    ax[1,1].set_ylim(ylim)
    ax[1,1].set_aspect('equal', 'box')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[:,1], shrink=.7, aspect=20)
    return fig,ax

fig8,ax8=ErrorFOV(DS1,DS2, figsize=(6,3),placement='right', maxDistance=pair_filter[2])
#fig8.tight_layout()
fig8.savefig(output_path+'fig8', transparent=True, dpi=dpi) 
