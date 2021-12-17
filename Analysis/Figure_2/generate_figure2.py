# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:07:57 2021

@author: Mels
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf
import copy
import time
import scipy.special as scpspc

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from dataset import dataset
from dataset_simulation import dataset_simulation, Affine_Deform
from CatmullRomSpline2D import CatmullRomSpline2D
plt.rc('font', size=30)

t=time.time()
#plt.close('all')
#%% fig1 - Gridview
def SplinesDeform(Dataset, gridsize=3000, edge_grids=1, error=10, random_error=None): # creating a splines offset error
    Dataset.edge_grids=edge_grids
    Dataset.gridsize=gridsize
    ControlPoints=Dataset.generate_CPgrid(gridsize, edge_grids)
    
    #ControlPoints+=rnd.randn(ControlPoints.shape[0],ControlPoints.shape[1],ControlPoints.shape[2])*error/gridsize
    ControlPoints=ControlPoints.numpy()
    ControlPoints[::2, ::2,:]+=error/gridsize
    ControlPoints[1::2, 1::2,:]-=error/gridsize
    if random_error is not None:
        ControlPoints+=np.random.rand(*(ControlPoints.shape))*random_error/gridsize
    ControlPoints=tf.Variable(ControlPoints, trainable=False, dtype=tf.float32)
    
    Dataset.SplinesModel=CatmullRomSpline2D(ControlPoints)
    Dataset.Apply_Splines()
    Dataset.SplinesModel=None
    Dataset.gridsize=None
    Dataset.edge_grids=None
    return Dataset


#%% loading the dataset
if False: ## Niekamp
    maxDistance=1000
    opt_fn=tf.optimizers.SGD
    DS01=dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
      pix_size=1, loc_error=1.4, mu=0.3,
      linked=False, FrameLinking=True, BatchOptimization=False)
    DS02=dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
      pix_size=1, loc_error=1.4, mu=0.3,
      linked=False, FrameLinking=True)
    DS01.load_dataset_excel()
    DS02.load_dataset_excel()
    DS01.pix_size=159
    DS02.pix_size=DS01.pix_size
    
    ## optimization params
    execute_linked=True
    learning_rate=1e-3
    epochs=300
    pair_filter=[250, 30, 30]
    gridsize=6500    
    
    DS01.link_dataset(maxDistance=maxDistance)
    DS01.AffineLLS()
    DS01.Filter(pair_filter[0]) 
    
    DS02.Apply_Affine(DS01.AffineMat)
    DS02.link_dataset(maxDistance=maxDistance)
    DS02.Filter(pair_filter[0]) 
    
    Num=DS01.ch1.pos.shape[0]
    output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_2/figure2_Niekamp/DataOutput/'
    GaussianFit=False
    

if False: #% Load FRET locs
    maxDistance=1000
    opt_fn=tf.optimizers.SGD
    DS01 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
                   'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
                  pix_size=159, loc_error=10, mu=0, coloc_error=np.sqrt(2)*(10),
                  imgshape=[256,512], linked=False, FrameLinking=True, BatchOptimization=False)
    DS01.load_dataset_hdf5(align_rcc=True)
        
    ## optimization params
    execute_linked=True
    learning_rate=2e-3
    epochs=300
    pair_filter=[250, 180, 180]
    gridsize=84
    
    DS01,DS02=DS01.SplitDatasetClusters()
    AffineClusters=DS01.AffineLLS_clusters()
    DS01.Apply_Affine(AffineClusters)
    DS01.link_dataset(maxDistance=maxDistance)
    DS01.AffineLLS()
    DS01.Filter(pair_filter[0]) 
    
    DS02.Apply_Affine(AffineClusters)
    DS02.Apply_Affine(DS01.AffineMat)
    DS02.link_dataset(maxDistance=maxDistance)
    DS02.Filter(pair_filter[0])
    
    Num=DS01.ch1.pos.shape[0]
    output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_2/figure2_Clusters/DataOutput/'
    GaussianFit=True
    

if True: #% Load FRET clusters
    maxDistance=300
    k=1
    opt_fn=tf.optimizers.SGD
    DS01 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
                   'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
                  pix_size=159, loc_error=10, mu=0, imgshape=[256,512], 
                  linked=False, FrameLinking=False, BatchOptimization=False, execute_linked=False)
    DS01.load_dataset_hdf5(align_rcc=True, transpose=False)
    DS01, DS02=DS01.SplitDatasetClusters()
    DS01clust=DS01.ClusterDataset(loc_error=None)
    DS02clust=DS02.ClusterDataset(loc_error=None)
    DS01clust.link_dataset(maxDistance=maxDistance)
    
    ## optimization params
    learning_rate1=2e-3
    epochs1=1000
    pair_filter1=[None, None]
    gridsize1=5000
    
    learning_rate=2e-3
    epochs=1000
    pair_filter=[None, None, maxDistance]
    gridsize=5000
    
    #% aligning clusters
    DS01clust.AffineLLS(maxDistance, k)
    DS01clust.Filter(pair_filter1[0]) 
    if epochs1 is not None:
        DS01clust.Train_Splines(learning_rate1, epochs1, gridsize1, edge_grids=1, opt_fn=tf.optimizers.SGD, 
                          maxDistance=maxDistance, k=k)
        DS01clust.Apply_Splines()
    DS01clust.Filter(pair_filter1[1])
    
    #% applying clusters
    DS01.copy_models(DS01clust) ## Copy all mapping parameters
    DS01.Apply_Affine(DS01clust.AffineMat)
    if DS01.SplinesModel is not None: DS01.Apply_Splines()
    if DS02clust is not None:
        DS02.copy_models(DS01clust) ## Copy all mapping parameters
        DS02.Apply_Affine(DS01clust.AffineMat)
        if DS02.SplinesModel is not None: DS02.Apply_Splines()
        
    #% linking dataset
    DS01.link_dataset(maxDistance=maxDistance)
    DS02.link_dataset(maxDistance=maxDistance)
        
    Num=DS01.ch1.pos.shape[0]
    output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_2/figure2_Clusters/DataOutput/'
    GaussianFit=False
    
    
if False: ## Grid
    ## dataset params
    opt_fn=tf.optimizers.SGD
    CRS_error=10 # implemented splines error
    random_error=3
    Num=14000     # number of points
    gridsize=3000
    deform_gridsize=1.5*gridsize
    loc_error=1.4
    DS01 = dataset_simulation(imgshape=[512, 512], loc_error=loc_error, linked=True,
                             pix_size=159, FrameLinking=False, BatchOptimization=False)
    deform=Affine_Deform(A=np.array([[ 1.0031357 ,  0.00181658, -1.3986971], 
                                  [-0.00123012,  0.9972918, 300.3556707 ]]))
    DS01.generate_dataset_grid(N=Num, deform=deform)
    DS01.ch2.pos.assign(tf.stack([DS01.ch2.pos[:,0]+400,DS01.ch2.pos[:,1]],axis=1))
    DS01=SplinesDeform(DS01, gridsize=deform_gridsize, error=CRS_error, random_error=random_error)
    DS01, DS02=DS01.SplitDataset()
    
    ## optimization params
    execute_linked=True
    learning_rate=1e-3
    epochs=300
    pair_filter = [None, None, 20]
    output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_2/figure2_Grid/DataOutput/'
    GaussianFit=True


#######################################################################################################################
#######################################################################################################################
#%% functions
def ErrorDistribution_r(DS1, simple_error=False, nbins=100, GaussianFit=True):
    if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')        
    dist, avg, r = DS1.ErrorDist(DS1.ch1.pos.numpy(), DS1.ch2.pos.numpy())
    
    if simple_error:
        return np.std(dist), np.average(dist)
    
    else:
        if GaussianFit:
            def func(r, sigma, mu):  # from Churchman et al 2006
                return np.exp(-(r - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2*np.pi)*sigma)
        else:
            def func(r, sigma):
                sigma2=sigma**2
                return (r/sigma2)*np.exp(-(.3**2+r**2)/2/sigma2)*scpspc.jv(0, r*.3/sigma2) # r/sigma2*np.exp(-r**2/2/sigma2)
            
        n_dists=np.histogram(dist, bins=nbins)
        N=DS1.ch1.pos.shape[0] * ( n_dists[1][1]-n_dists[1][0] )
        xn=(n_dists[1][:-1]+n_dists[1][1:])/2
        if GaussianFit: popt,pcov=curve_fit(func, xn, n_dists[0]/N, p0=[np.std(xn), np.average(xn)])
        else: popt,pcov=curve_fit(func, xn, n_dists[0]/N, p0=np.std(xn))
        if GaussianFit: return popt
        else: return pcov, popt 


#%% fig2 - Error vs Epochs opts
opts=[ tf.optimizers.Adam, tf.optimizers.Adagrad, tf.optimizers.Adadelta, tf.optimizers.Adamax,
      tf.optimizers.Ftrl, tf.optimizers.Nadam, tf.optimizers.RMSprop, tf.optimizers.SGD ]
opts_name=[ 'Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
epochs_fig2=np.logspace(0,3,20).astype('int')
epochs_fig2=np.concatenate([[1],epochs_fig2])

popt0 = ErrorDistribution_r(DS01, GaussianFit=GaussianFit)
popt1 = ErrorDistribution_r(DS02, GaussianFit=GaussianFit)
sigma1_fig2, sigma2_fig2=(popt0[0]*np.ones([len(opts), len(epochs_fig2)]),popt1[0]*np.ones([len(opts), len(epochs_fig2)]))
mu1_fig2,mu2_fig2=(popt0[1]*np.ones([len(opts), len(epochs_fig2)]),popt1[1]*np.ones([len(opts), len(epochs_fig2)]))

for i in range(len(opts)):
    for j in range(1,len(epochs_fig2)):
        DS1=copy.deepcopy(DS01)
        DS2=copy.deepcopy(DS02)
        try:
            DS1.Train_Splines(learning_rate, gridsize=gridsize, edge_grids=1, 
                              epochs=epochs_fig2[j], opt_fn=opts[i])
            DS1.Apply_Splines() 
            DS2.copy_models(DS1)
            DS2.Apply_Splines() 
            
            DS1.Filter(pair_filter[1])
            popt = ErrorDistribution_r(DS1, GaussianFit=GaussianFit)
            sigma1_fig2[i,j]=(popt[0])
            mu1_fig2[i,j]=(popt[1])
            
            DS2.Filter(pair_filter[1])
            popt = ErrorDistribution_r(DS2, GaussianFit=GaussianFit)
            sigma2_fig2[i,j]=(popt[0])
            mu2_fig2[i,j]=(popt[1])
        except:
            sigma1_fig2[i,j]=(0)
            mu1_fig2[i,j]=(np.nan)
            sigma2_fig2[i,j]=(0)
            mu2_fig2[i,j]=(np.nan)
        del DS1, DS2


np.savetxt(output_path+'sigma1_fig2.txt',sigma1_fig2)
np.savetxt(output_path+'mu1_fig2.txt',mu1_fig2)
np.savetxt(output_path+'sigma2_fig2.txt',sigma2_fig2)
np.savetxt(output_path+'mu2_fig2.txt',mu2_fig2)
np.savetxt(output_path+'epochs_fig2.txt',epochs_fig2)


#######################################################################################################################
#######################################################################################################################
#%% fig3 - Error vs Learning-rates 
NN=20
learning_rates = np.logspace(-7, 0, NN)
sigma1_fig3,sigma2_fig3=(np.zeros(NN, dtype=np.float32),np.zeros(NN, dtype=np.float32))
std1_fig3,std2_fig3=(np.zeros(NN, dtype=np.float32),np.zeros(NN, dtype=np.float32))

for i in range(NN):
    DS1=copy.deepcopy(DS01)
    DS2=copy.deepcopy(DS02)
        
    try:
        DS1.Train_Splines(learning_rates[i], epochs, gridsize, edge_grids=1, opt_fn=opt_fn)
        DS1.Apply_Splines() 
        DS2.copy_models(DS1)
        DS2.Apply_Splines() 
        
        DS1.Filter(pair_filter[1])
        popt = ErrorDistribution_r(DS1, GaussianFit=GaussianFit)
        sigma1_fig3[i]=popt[1]
        std1_fig3[i]=popt[0]
                    
        DS2.Filter(pair_filter[1])
        popt = ErrorDistribution_r(DS2, GaussianFit=GaussianFit)
        sigma2_fig3[i]=popt[1]
        std2_fig3[i]=popt[0]
    except:
        sigma1_fig3[i]=np.nan
        sigma2_fig3[i]=np.nan
        std1_fig3[i]=0
        std2_fig3[i]=0
    del DS1,DS2


np.savetxt(output_path+'sigma1_fig3.txt',sigma1_fig3)
np.savetxt(output_path+'sigma2_fig3.txt',sigma2_fig3)
np.savetxt(output_path+'learning_rates.txt',learning_rates)
np.savetxt(output_path+'std1_fig3.txt',std1_fig3)
np.savetxt(output_path+'std2_fig3.txt',std2_fig3)

#######################################################################################################################
#######################################################################################################################
#%% fig4 - Error vs Gridsize
begin=60
end=13000
N=10#50
gridsizes=np.logspace(np.log10(begin), np.log10(end), N, endpoint=False)
sigma1_fig4,sigma2_fig4=(np.zeros(len(gridsizes),dtype=float),np.zeros(len(gridsizes),dtype=float))
std1_fig4,std2_fig4=(np.zeros(len(gridsizes),dtype=float),np.zeros(len(gridsizes),dtype=float))

for i in range(len(gridsizes)):
    DS1=copy.deepcopy(DS01)
    DS2=copy.deepcopy(DS02)
    #try:         
    DS1.Train_Splines(learning_rate, epochs, gridsizes[i], edge_grids=1, opt_fn=opt_fn)
    DS1.Apply_Splines() 
    DS2.copy_models(DS1)
    DS2.Apply_Splines()
            
    DS1.Filter(pair_filter[1])
    popt=ErrorDistribution_r(DS1, GaussianFit=GaussianFit)
    sigma1_fig4[i]=popt[1]
    std1_fig4[i]=popt[0]
    
    DS2.Filter(pair_filter[1])
    popt=ErrorDistribution_r(DS2, GaussianFit=GaussianFit)
    sigma2_fig4[i]=popt[1]
    std2_fig4[i]=popt[0]
    '''
    except:
        sigma1_fig4[i]=100
        sigma2_fig4[i]=100
        std1_fig4[i]=0
        std2_fig4[i]=0
    '''
    del DS1, DS2


np.savetxt(output_path+'sigma1_fig4.txt',sigma1_fig4)
np.savetxt(output_path+'sigma2_fig4.txt',sigma2_fig4)
np.savetxt(output_path+'std1_fig4.txt',std1_fig4)
np.savetxt(output_path+'std2_fig4.txt',std2_fig4)
np.savetxt(output_path+'gridsizes.txt', gridsizes)
 
#%%
print('DONE IN '+str((time.time()-t)//60)+' MINUTES AND '+str(round((time.time()-t)%60,0))+' SECONDS!')