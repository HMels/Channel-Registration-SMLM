# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 17:20:52 2021

@author: Mels
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import numpy as np
from scipy.optimize import curve_fit
import scipy.special as scpspc
import pandas as pd
from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')
output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_Misc/Figures/'

from dataset import dataset
from Channel import Channel

plt.rc('font', size=10)
dpi=800

plt.close('all')
#%% fn
def ErrorDistribution_r(DS1, fig, ax, nbins=30, xlim=31, error=None, mu=.3, fit_data=True, fit_gaus=False):
    if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    dist, avg, r = DS1.ErrorDist(DS1.ch1.pos_all(), DS1.ch2.pos_all())
    dist=dist[np.argwhere(dist<xlim)]
        
    n=ax.hist(dist, range=[0,xlim], alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)#, label='N='+str(DS1.ch1.pos.shape[0]))
    ymax = np.max(n[0])*1.1
    if fit_data: ## fit bar plot data using curve_fit
        if fit_data: ## fit bar plot data using curve_fit
            def func(r, mu, sigma):
                return np.exp(-(r - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2*np.pi)*sigma)
        else:   
            def func(r, sigma, mu): # from Churchman et al 2006
                sigma2=sigma**2
                return (r/sigma2)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)

        N = DS1.ch1.pos.shape[0] * ( n[1][1]-n[1][0] )
        xn=(n[1][:-1]+n[1][1:])/2 
        popt, pcov = curve_fit(func, xn, n[0]/N, p0=[np.std(xn), np.average(xn)])
        x = np.linspace(0, xlim, 1000)
        y = func(x, *popt)*N
        ax.plot(x, y, c='g',label=(r'$\sigma$='+str(round(popt[0],2))+'nm\n$\mu$='+str(round(popt[1],2))+'nm'))
    
        if error is not None: ## plot how function should look like
            sgm=np.sqrt(2)*error
            y = func(x, sgm, mu)*N
            ax.plot(x, y, c='b')#,label=(r'optimum: $\mu$='+str(round(mu,2))+', $\sigma$='+str(round(sgm,2))+'nm'))
            if np.max(y)>ymax: ymax=np.max(y)*1.1

    # Some extra plotting parameters
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xlim])
    #ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

#%% fig1
maxDistance=1000
DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
  pix_size=1, loc_error=1.4, mu=0.3,
  linked=False, FrameLinking=True, BatchOptimization=False)
#DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
#  pix_size=1, loc_error=1.4, mu=0.3,
#  linked=False, FrameLinking=True)
DS1.load_dataset_excel()
#DS2.load_dataset_excel()
DS1.pix_size=159
#DS2.pix_size=DS1.pix_size

## optimization params
pair_filter = [30, 30, 30]
gridsize=6500

## plotting params
nbins=100
DS1.link_dataset(maxDistance=maxDistance)
#DS2.link_dataset(maxDistance=maxDistance)

fig1=plt.figure(figsize=(4,2))
ax1=fig1.add_subplot(111)
fig1, ax1=ErrorDistribution_r(DS1, fig1, ax1, nbins=nbins, xlim=maxDistance, fit_gaus=False)
ylim=ax1.get_ylim()[1]
ax1.text(x=maxDistance*.03,y=ylim*.7, s=r'$original$', color='black',ha='left', va='top')
ax1.text(x=maxDistance*.03,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='left', va='top')
ax1.text(x=maxDistance*.03,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='left', va='top')
fig1.savefig(output_path+'AffineLLS_fig1', transparent=True, dpi=dpi)


#% fig2
DS1.AffineLLS()
DS1.Filter(pair_filter[0]) 
fig1=plt.figure(figsize=(4,2))
ax1=fig1.add_subplot(111)
fig1, ax1=ErrorDistribution_r(DS1, fig1, ax1, nbins=nbins, xlim=pair_filter[0])
ylim=ax1.get_ylim()[1]
ax1.text(x=pair_filter[0]*.97,y=ylim*.7, s=r'$AffineLLS$', color='black',ha='right', va='top')
ax1.text(x=pair_filter[0]*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax1.text(x=pair_filter[0]*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig1.savefig(output_path+'AffineLLS_fig2', transparent=True, dpi=dpi)


#%% fig3
maxDistance=300
k=8
DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
               'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
              pix_size=159, loc_error=10, mu=0, imgshape=[256,512], 
              linked=False, FrameLinking=True, BatchOptimization=False, execute_linked=False)
DS1.load_dataset_hdf5(align_rcc=True, transpose=False)
DS1clust=DS1.ClusterDataset(loc_error=None)
DS1clust.execute_linked=True
## optimization params
pair_filter=[maxDistance, maxDistance, maxDistance]
gridsize=7500

DS1clust.link_dataset(maxDistance=maxDistance)
fig1=plt.figure(figsize=(4,2))
ax1=fig1.add_subplot(111)
fig1, ax1=ErrorDistribution_r(DS1clust, fig1, ax1, nbins=nbins, xlim=maxDistance, fit_gaus=False)
ylim=ax1.get_ylim()[1]
#ax1.set_xticks([])
ax1.text(x=maxDistance*.97,y=ylim*.7, s=r'$original$', color='black',ha='right', va='top')
ax1.text(x=maxDistance*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax1.text(x=maxDistance*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig1.savefig(output_path+'AffineLLS_fig3', transparent=True, dpi=dpi)


#%fig4
#% aligning clusters
DS1clust.AffineLLS(maxDistance, k)
fig1=plt.figure(figsize=(4,2))
ax1=fig1.add_subplot(111)
fig1, ax1=ErrorDistribution_r(DS1clust, fig1, ax1, nbins=nbins, xlim=pair_filter[0])
ylim=ax1.get_ylim()[1]
#ax1.set_xticks([])
ax1.text(x=pair_filter[0]*.97,y=ylim*.7, s=r'$AffineLLS$', color='black',ha='right', va='top')
ax1.text(x=pair_filter[0]*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax1.text(x=pair_filter[0]*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig1.savefig(output_path+'AffineLLS_fig4', transparent=True, dpi=dpi)


#%% fig5
'''
#% applying clusters
DS1.copy_models(DS1clust) ## Copy all mapping parameters
DS1.Apply_Affine(DS1clust.AffineMat)
if DS1.SplinesModel is not None: DS1.Apply_Splines()
    
#% linking dataset
if not DS1.Neighbours: DS1.kNearestNeighbour(k=k, maxDistance=maxDistance)
DS1copy=copy.deepcopy(DS1)
DS1copy.link_dataset(maxDistance=maxDistance)
fig1=plt.figure(figsize=(4,2))
ax1=fig1.add_subplot(111)
fig1, ax1=ErrorDistribution_r(DS1copy, fig1, ax1, nbins=nbins, xlim=maxDistance, fit_gaus=False)
ylim=ax1.get_ylim()[1]
ax1.text(x=maxDistance*.97,y=ylim*.7, s=r'$original$', color='black',ha='right', va='top')
ax1.text(x=maxDistance*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax1.text(x=maxDistance*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig1.savefig(output_path+'AffineLLS_fig5', transparent=True, dpi=dpi)


#% fig6
DS1.AffineLLS(maxDistance, k)
DS1.link_dataset(maxDistance=maxDistance)
DS1.Filter(pair_filter[0]) 
fig1=plt.figure(figsize=(4,2))
ax1=fig1.add_subplot(111)
fig1, ax1=ErrorDistribution_r(DS1, fig1, ax1, nbins=nbins, xlim=pair_filter[0])
ylim=ax1.get_ylim()[1]
ax1.text(x=pair_filter[0]*.97,y=ylim*.7, s=r'$AffineLLS$', color='black',ha='right', va='top')
ax1.text(x=pair_filter[0]*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax1.text(x=pair_filter[0]*.97,y=ylim*.6, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig1.savefig(output_path+'AffineLLS_fig6', transparent=True, dpi=dpi)
'''