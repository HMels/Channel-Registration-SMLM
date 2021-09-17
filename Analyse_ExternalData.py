# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:46:47 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

       
#%% functions
def load_dataset(path):
    data = pd.read_csv(path)
    pair = np.array(data[['X1','Y1','X2','Y2', 'Distance']])
    
    average = np.average(pair[:,:2], axis=0)
    pair[:,:2]-=average
    pair[:,2:4]-=average
    return pair


def ErrorDistribution(pair, nbins=30):
# just plots the error distribution after mapping
    plt.figure()
    dist=pair[:,4]
    avg1 = np.average(dist)
    n1 = plt.hist(dist, label=('Mapped Error = '+str(round(avg1,2))+'nm'), alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
    ymax = np.max(n1[0]*1.1)
    #plt.title('Zoomed in on Mapping Error')
    plt.ylim([0,ymax])
    plt.xlim(0,31)
    plt.xlabel('distance [nm]')
    plt.ylabel('# of localizations')
    plt.legend()
    plt.tight_layout()
    
    
def ErrorDistribution_xy(pair, nbins=30):
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    distx=pair[:,0]-pair[:,2]
    avgx = np.average(distx)
    stdx = np.std(distx)
    disty=pair[:,1]-pair[:,3]
    avgy = np.average(disty)
    stdy = np.std(disty)
    
    nx = ax[0].hist(distx, label=(r'$\mu$ = '+str(round(avgx,2))+'nm, $\sigma$ = '+str(round(stdx,2))+'nm'),
                    alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
    ny = ax[1].hist(disty, label=(r'$\mu$ = '+str(round(avgy,2))+'nm, $\sigma$ = '+str(round(stdy,2))+'nm'),
                    alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
    
    ymax = np.max([np.max(nx[0]),np.max(ny[0])])*1.1
    ax[0].set_ylim([0,ymax])
    ax[0].set_xlim(-31,31)
    ax[0].set_xlabel('x-distance [nm]')
    ax[0].set_ylabel('# of localizations')
    ax[0].legend()
    
    ax[1].set_ylim([0,ymax])
    ax[1].set_xlim(-31,31)
    ax[1].set_xlabel('y-distance [nm]')
    ax[1].set_ylabel('# of localizations')
    ax[1].legend()
    fig.tight_layout()
    

#%% plotting the error in a [x1, x2] plot like in the paper        
def ErrorPlotImage(pair, pair1, maxDist=30, ps=5, cmap='seismic'):
    
    dist=np.stack((pair[:,0]-pair[:,2], pair[:,1]-pair[:,3]), axis=1)
    dist1=np.stack((pair1[:,0]-pair1[:,2], pair1[:,1]-pair1[:,3]), axis=1)
    pos11=pair[:,:2]/1000
    pos21=pair1[:,:2]/1000
    
    vmin=np.min((np.min(dist[:,0]),np.min(dist1[:,0]),np.min(dist[:,1]),np.min(dist1[:,1])))
    vmax=np.max((np.max(dist[:,0]),np.max(dist1[:,0]),np.max(dist[:,1]),np.max(dist1[:,1])))
    
    
    fig, ax = plt.subplots(2,2)
    ax[0][0].scatter(pos11[:,0], pos11[:,1], s=ps, c=dist[:,0], cmap=cmap, vmin=vmin, vmax=vmax)
    #ax[0][0].set_xlabel('x-position [\u03bcm]')
    ax[0][0].set_ylabel('Set 1 Fiducials\ny-position [\u03bcm]')
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-offset [nm]', ax=ax[0][0])
    ax[0][0].set_aspect('equal', 'box')
    
    ax[0][1].scatter(pos11[:,0], pos11[:,1], s=ps, c=dist[:,1], cmap=cmap, vmin=vmin, vmax=vmax)
    #ax[0][1].set_xlabel('x-position [\u03bcm]')
    #ax[0][1].set_ylabel('y-position [\u03bcm]')
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-offset [nm]', ax=ax[0][1])
    ax[0][1].set_aspect('equal', 'box')

    ax[1][0].scatter(pos21[:,0], pos21[:,1], s=ps, c=dist1[:,0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1][0].set_xlabel('x-position [\u03bcm]')
    ax[1][0].set_ylabel('Set 2 Fiducials\ny-position [\u03bcm]')
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-offset [nm]', ax=ax[1][0])
    ax[1][0].set_aspect('equal', 'box')
    
    ax[1][1].scatter(pos21[:,0], pos21[:,1], s=ps, c=dist1[:,1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1][1].set_xlabel('x-position [\u03bcm]')
    #ax[1][1].set_ylabel('y-position [\u03bcm]')
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-offset [nm]', ax=ax[1][1])
    ax[1][1].set_aspect('equal', 'box')
    fig.tight_layout()
    
    
#%% Loading Dataset

fiducials1 = load_dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration_After_Mapping/Set1/set1_beads_locs_NoLocal.csv')
fiducials2 = load_dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration_After_Mapping/Set2/set2_beads_locs_NoLocal.csv')

#fiducials1 = load_dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration_After_Mapping/Set1/set1_beads_locs.csv')
#fiducials2 = load_dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration_After_Mapping/Set2/set2_beads_locs.csv')


#%% output
plt.close('all')
nbins=100

## DS1
ErrorDistribution_xy(fiducials1, nbins=nbins)

## DS2
ErrorDistribution_xy(fiducials2, nbins=nbins)

## DS1 vs DS2
ErrorPlotImage(fiducials1, fiducials2)

