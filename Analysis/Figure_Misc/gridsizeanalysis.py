# main.py
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
plt.close('all')

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')
from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_Misc/'
dpi=450
begin=60
end=25000
N=10
gridsizes=np.logspace(np.log10(begin), np.log10(end), N, endpoint=True)
#%%
if True:
    np.savetxt(output_path+'DataOutput/sigmax1.txt',[])
    np.savetxt(output_path+'DataOutput/sigmay1.txt',[])
    np.savetxt(output_path+'DataOutput/sigmax2.txt',[])
    np.savetxt(output_path+'DataOutput/sigmay2.txt',[])
    np.savetxt(output_path+'DataOutput/sigma1.txt',[])
    np.savetxt(output_path+'DataOutput/sigma2.txt',[])
    np.savetxt(output_path+'DataOutput/covx1.txt', [])
    np.savetxt(output_path+'DataOutput/covy1.txt', [])
    np.savetxt(output_path+'DataOutput/covx2.txt', [])
    np.savetxt(output_path+'DataOutput/covy2.txt', [])
    np.savetxt(output_path+'DataOutput/cov1.txt', [])
    np.savetxt(output_path+'DataOutput/cov2.txt', [])
    np.savetxt(output_path+'DataOutput/gridsizes.txt',gridsizes)


#%%
for gridsize in gridsizes:
    print('####################################################################')
    print('GRIDSIZE_i=',gridsize,'from',gridsizes)
    DS2=None
      
    if True: #% Load FRET clusters
        maxDistance=300
        k=8
        DS1 = dataset(#['C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan1_picked.hdf5',
                      # 'C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan2_picked.hdf5'],
                      ['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
                       'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
                      pix_size=159, loc_error=10, mu=0, imgshape=[256,512], 
                      linked=False, FrameLinking=True, BatchOptimization=False, execute_linked=False)
        DS1.load_dataset_hdf5(align_rcc=True, transpose=False)
        DS1, DS2=DS1.SplitDatasetClusters()
        DS1clust=DS1.ClusterDataset(loc_error=None)
        DS1clust.execute_linked=True
        DS2clust=DS2.ClusterDataset(loc_error=None)
        DS1clust.link_dataset(maxDistance=maxDistance)
        
        ## optimization params
        learning_rate=5e-5
        epochs=300
        pair_filter=[None, None, maxDistance]
        
        #% aligning clusters
        DS1clust.AffineLLS(maxDistance, k)
        
        #% applying clusters
        DS1.copy_models(DS1clust) ## Copy all mapping parameters
        DS1.Apply_Affine(DS1clust.AffineMat)
        DS2.copy_models(DS1clust) ## Copy all mapping parameters
        DS2.Apply_Affine(DS1clust.AffineMat)
        DS1.Filter(pair_filter[0]) 
            
        #% linking dataset
        if not DS1.linked: DS1.link_dataset(maxDistance=maxDistance)
        
        
    #%% running the CatmullRomSplines
    start=time.time()
    if epochs is not None:
        DS1.execute_linked=True
        if not DS1.linked: DS1.link_dataset(maxDistance=maxDistance)
        DS1.Train_Splines(learning_rate, epochs, gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD, 
                          maxDistance=maxDistance, k=k)
        DS1.Apply_Splines()
        
    DS1.Filter(pair_filter[1])
    print('Optimized in',round((time.time()-start)/60,1),'minutes!')
    
    if DS2 is not None:
        DS2.copy_models(DS1) ## Copy all mapping parameters
        if DS2.SplinesModel is not None: DS2.Apply_Splines()
       
    
    #%% output
    nbins=100
    xlim=pair_filter[2]
        
    if not DS1.linked:
        DS1.link_dataset(maxDistance=maxDistance)
    
    ## DS1
    poptx1,popty1, pcovx1, pcovy1=DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.coloc_error, fit_data=True)
    popt1, pcov1=DS1.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1.coloc_error, mu=DS1.mu, fit_data=True)
    
    
    #%% DS2 output
    if DS2 is not None:
        if not DS2.linked: 
            DS2.link_dataset(maxDistance=maxDistance)
                
        DS2.Filter(pair_filter[1])
        poptx2,popty2, pcovx2, pcovy2=DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS2.coloc_error)
        popt2, pcov2=DS2.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2.coloc_error, mu=DS2.mu)
    
    #%% save it
    sigmax1=np.loadtxt(output_path+'DataOutput/sigmax1.txt')
    sigmay1=np.loadtxt(output_path+'DataOutput/sigmay1.txt')
    sigmax2=np.loadtxt(output_path+'DataOutput/sigmax2.txt')
    sigmay2=np.loadtxt(output_path+'DataOutput/sigmay2.txt')
    sigma1=np.loadtxt(output_path+'DataOutput/sigma1.txt')
    sigma2=np.loadtxt(output_path+'DataOutput/sigma2.txt')
    covx1=np.loadtxt(output_path+'DataOutput/covx1.txt')
    covy1=np.loadtxt(output_path+'DataOutput/covy1.txt')
    covx2=np.loadtxt(output_path+'DataOutput/covx2.txt')
    covy2=np.loadtxt(output_path+'DataOutput/covy2.txt')
    cov1=np.loadtxt(output_path+'DataOutput/cov1.txt')
    cov2=np.loadtxt(output_path+'DataOutput/cov2.txt')

    sigmax1=np.append(sigmax1,poptx1)
    sigmay1=np.append(sigmay1,popty1)
    sigmax2=np.append(sigmax2,poptx2)
    sigmay2=np.append(sigmay2,popty2)
    sigma1=np.append(sigma1,popt1)
    sigma2=np.append(sigma2,popt2)
    covx1=np.append(covx1,[pcovx1[0,0], pcovx1[1,1]])
    covy1=np.append(covy1,[pcovy1[0,0], pcovy1[1,1]])
    covx2=np.append(covx2,[pcovx2[0,0], pcovx2[1,1]])
    covy2=np.append(covy2,[pcovy2[0,0], pcovy2[1,1]])
    cov1=np.append(cov1,[pcov1[0,0], pcov1[1,1]])
    cov2=np.append(cov2,[pcov2[0,0], pcov2[1,1]])
    
    np.savetxt(output_path+'DataOutput/sigmax1.txt',sigmax1)
    np.savetxt(output_path+'DataOutput/sigmay1.txt',sigmay1)
    np.savetxt(output_path+'DataOutput/sigmax2.txt',sigmax2)
    np.savetxt(output_path+'DataOutput/sigmay2.txt',sigmay2)
    np.savetxt(output_path+'DataOutput/sigma1.txt',sigma1)
    np.savetxt(output_path+'DataOutput/sigma2.txt',sigma2)
    np.savetxt(output_path+'DataOutput/covx1.txt', covx1)
    np.savetxt(output_path+'DataOutput/covy1.txt', covy1)
    np.savetxt(output_path+'DataOutput/covx2.txt', covx2)
    np.savetxt(output_path+'DataOutput/covy2.txt', covy2)
    np.savetxt(output_path+'DataOutput/cov1.txt', cov1)
    np.savetxt(output_path+'DataOutput/cov2.txt', cov2)
    
    plt.close('all')
    
    
#%% plotting it
gridsizes=np.loadtxt(output_path+'DataOutput/gridsizes.txt')
sigmax1=np.loadtxt(output_path+'DataOutput/sigmax1.txt')
sigmay1=np.loadtxt(output_path+'DataOutput/sigmay1.txt')
sigmax2=np.loadtxt(output_path+'DataOutput/sigmax2.txt')
sigmay2=np.loadtxt(output_path+'DataOutput/sigmay2.txt')
covx1=np.loadtxt(output_path+'DataOutput/covx1.txt')
covy1=np.loadtxt(output_path+'DataOutput/covy1.txt')
covx2=np.loadtxt(output_path+'DataOutput/covx2.txt')
covy2=np.loadtxt(output_path+'DataOutput/covy2.txt')

fig=plt.figure(figsize=(6,4))
lw=1

ax1=fig.add_subplot(211)
ax1.plot(gridsizes, sigmax1[1::2], '--', lw=lw, color='blue', label='estimation x-error')
ax1.plot(gridsizes, sigmax2[1::2], '--', lw=lw, color='red', label='testing x-error')
ax1.plot(gridsizes, sigmay1[1::2], ':', lw=lw, color='blue', label='estimation y-error')
ax1.plot(gridsizes, sigmay2[1::2], ':', lw=lw, color='red', label='testing y-error')
ax1.set_xscale('log')
ax1.set_ylabel('precision')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])
ax1.axes.get_xaxis().set_visible(False)
ax1.set_xlim((gridsizes[0], gridsizes[-1]))
ax1.legend(frameon=False, loc='upper left')


ax2=fig.add_subplot(212)
ax2.plot(gridsizes, np.abs(sigmax1[::2]), '--', lw=lw, color='blue', label='estimation x-error')
ax2.plot(gridsizes, np.abs(sigmax2[::2]), '--', lw=lw, color='red', label='testing x-error')
ax2.plot(gridsizes, np.abs(sigmay1[::2]), ':', lw=lw, color='blue', label='estimation y-error')
ax2.plot(gridsizes, np.abs(sigmay2[::2]), ':', lw=lw, color='red', label='testing y-error')
ax2.set_xscale('log')
ax2.set_ylabel('bias')
ax2.set_xlabel('gridsize [nm]')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim((gridsizes[0], gridsizes[-1]))

fig.tight_layout()
fig.savefig(output_path+'Figures/gridsizesxy', transparent=True, dpi=dpi)


#%%
gridsizes=np.loadtxt(output_path+'DataOutput_backup/gridsizes.txt')
sigma1=np.loadtxt(output_path+'DataOutput_backup/sigma1.txt')
sigma2=np.loadtxt(output_path+'DataOutput_backup/sigma2.txt')
cov1=np.loadtxt(output_path+'DataOutput_backup/cov1.txt')
cov2=np.loadtxt(output_path+'DataOutput_backup/cov2.txt')

fig=plt.figure(figsize=(4.35,3.83))
lw=1

ax1=fig.add_subplot(211)
ax1.errorbar(gridsizes, np.abs(sigma1[::2]), yerr=cov1[::2], lw=lw, color='blue', label='estimation')
ax1.errorbar(gridsizes, np.abs(sigma2[::2]), yerr=cov2[::2], lw=lw, color='red', label='testing')
ax1.set_ylabel('precision')
ax1.set_xscale('log')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.axes.get_xaxis().set_visible(False)
ax1.set_xticks([])
ax1.set_xlim((gridsizes[0], gridsizes[-1]))
ax1.legend(frameon=False, loc='lower right')


ax2=fig.add_subplot(212)
ax2.errorbar(gridsizes, np.abs(sigma1[1::2]), yerr=cov1[1::2], lw=lw, color='blue', label='estimation')
ax2.errorbar(gridsizes, np.abs(sigma2[1::2]), yerr=cov2[1::2], lw=lw, color='red', label='testing')
ax2.set_xscale('log')
ax2.set_ylabel('bias')
ax2.set_xlabel('gridsize [nm]')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim((gridsizes[0], gridsizes[-1]))
fig.tight_layout()

fig.savefig(output_path+'Figures/gridsizes', transparent=True, dpi=dpi)