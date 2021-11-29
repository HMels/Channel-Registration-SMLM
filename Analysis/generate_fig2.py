# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:07:57 2021

@author: Mels
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import time
import scipy.special as scpspc

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
import time
from Channel import Channel
from Align_Modules.Splines import CatmullRomSpline2D
from Align_Modules.Shift import ShiftModel
plt.rc('font', size=17)

t=time.time()
plt.close('all')
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
    Dataset.ch2.pos.assign(Dataset.InputSplines(
                    Dataset.Transform_Model(Dataset.SplinesModel, ch2=Dataset.InputSplines(Dataset.ch2.pos)),
                    inverse=True))
    Dataset.SplinesModel=None
    Dataset.gridsize=None
    Dataset.edge_grids=None
    return Dataset


DS2=None
if False: ## Niekamp
    ## dataset params
    loc_error=1.4
    N_it=1
    DS0 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  pix_size=1, loc_error=loc_error, mu=0.3, coloc_error=np.sqrt(2)*1.4,
                  linked=False, FrameLinking=True, BatchOptimization=False)
    DS0.load_dataset_excel()
    DS0.pix_size=159
    DS0.link_dataset(maxDistance=1000)
    Num=DS0.ch1.pos.shape[0]
    
    ## optimization params
    execute_linked=True
    learning_rates = [1e3, .1, 1e-3]
    epochs = [100, None, 300]
    pair_filter = [250, 30, 20]
    gridsize=6500
    figsize1=(16.5,7)


if True: ## Grid
    ## dataset params
    CRS_error=10 # implemented splines error
    random_error=3
    Num=14000     # number of points
    deform_gridsize=1.5*gridsize
    loc_error=1.4
    N_it=1
    DS0 = dataset_simulation(imgshape=[512, 512], loc_error=loc_error, linked=True,
                             pix_size=159, FrameLinking=False, BatchOptimization=False)
    deform=Affine_Deform(A=np.array([[ 1.0031357 ,  0.00181658, -1.3986971], 
                                  [-0.00123012,  0.9972918, 300.3556707 ]]))
    DS0.generate_dataset_grid(N=Num, deform=deform)
    DS0.ch2.pos.assign(tf.stack([DS0.ch2.pos[:,0]+400,DS0.ch2.pos[:,1]],axis=1))
    DS0=SplinesDeform(DS0, gridsize=deform_gridsize, error=CRS_error, random_error=random_error)
    
    ## optimization params
    execute_linked=True
    learning_rates = [1e3, .1, 1e-3]
    epochs = [100, None, 300]
    pair_filter = [None, None, 20]
    gridsize=3000
    figsize1=(16.5,7)


if False: #% Load FRET clusters
    ## dataset params
    CRS_error=10 # implemented splines error
    random_error=3
    Num=73308     # number of points
    deform_gridsize=1.5*gridsize
    loc_error=10
    N_it=1
    DS0 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5', 
                   'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5'],
                  pix_size=159, loc_error=loc_error, mu=0, coloc_error=np.sqrt(2)*loc_error,
                  imgshape=[256,512], linked=False, FrameLinking=True, BatchOptimization=False)
    DS0.load_dataset_hdf5(align_rcc=False)  
    DS0.link_dataset(maxDistance=800)
    
    ## optimization params
    execute_linked=True
    learning_rates = [1000, .1, 2e-3]
    epochs = [100, None, 200]
    pair_filter = [250, 250, 250]
    gridsize=100
    figsize1=(9.5,7)
    

fig,ax=DS0.ErrorFOV(figsize=figsize1)
ax[1].text(x=np.max(DS0.ch2.pos[:,0])/1000-2, y=np.max(DS0.ch2.pos[:,1])/1000-2, 
           s='N='+str(Num), ha='right', va='top', bbox=dict(boxstyle="square",
                                                           ec=(1., 0.5, 0.5),
                                                           fc=(1., 0.8, 0.8),
                                                           ))


#######################################################################################################################
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
        return popt
            

## ShiftModel 
DS0shift=copy.deepcopy(DS0)
DS0shift.ShiftModel=ShiftModel()
DS0shift.Train_Model(DS0shift.ShiftModel, lr=learning_rates[0], epochs=epochs[0], opt_fn=tf.optimizers.Adagrad)
DS0shift.Transform_Model(DS0shift.ShiftModel)
DS0shift.Filter(pair_filter[0])


#%% fig2 - Error vs Epochs opts
opts=[ tf.optimizers.Adam, tf.optimizers.Adagrad, tf.optimizers.Adadelta, tf.optimizers.Adamax,
      tf.optimizers.Ftrl, tf.optimizers.Nadam, tf.optimizers.RMSprop, tf.optimizers.SGD ]
opts_name=[ 'Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', '']

epochs_fig2 = 2*np.logspace(0,3,7).astype('int')

popt0 = ErrorDistribution_r(DS0)
popt1 = ErrorDistribution_r(DS0shift)
sigma1_fig2, sigma2_fig2=(popt0[0]*np.ones([len(opts), len(epochs_fig2)]),popt1[0]*np.ones([len(opts), len(epochs_fig2)]))
mu1_fig2,mu2_fig2=(popt0[1]*np.ones([len(opts), len(epochs_fig2)]),popt1[1]*np.ones([len(opts), len(epochs_fig2)]))

#% Shift
for i in range(len(opts)):
    for j in range(len(epochs_fig2)):
        DS1=copy.deepcopy(DS0)        
        try:
            DS1.ShiftModel=ShiftModel()
            DS1.Train_Model(DS1.ShiftModel, lr=learning_rates[0], epochs=epochs_fig2[j], opt_fn=opts[i])
            DS1.Transform_Model(DS1.ShiftModel)
            popt = ErrorDistribution_r(DS1)
            sigma1_fig2[i,j]=(popt[0])
            mu1_fig2[i,j]=(popt[1])
        except:
            sigma1_fig2[i,j]=(0)
            mu1_fig2[i,j]=(np.nan)
        del DS1

#% Spline
for i in range(len(opts)):
    for j in range(len(epochs_fig2)):
        DS2=copy.deepcopy(DS0shift)
        try:
            ch1_input,ch2_input=DS2.InitializeSplines(gridsize=gridsize, edge_grids=1)
            DS2.SplinesModel=CatmullRomSpline2D(DS2.ControlPoints)
            DS2.Train_Model(DS2.SplinesModel, lr=learning_rates[2], epochs=epochs_fig2[j], opt_fn=opts[i], 
                             ch1=ch1_input, ch2=ch2_input)  
            
            # applying the model
            DS2.ControlPoints = DS2.SplinesModel.ControlPoints
            DS2.ch2.pos.assign(DS2.InputSplines(
                DS2.Transform_Model(DS2.SplinesModel, ch2=DS2.InputSplines(DS2.ch2.pos)),
                inverse=True))
            popt = ErrorDistribution_r(DS2)
            sigma2_fig2[i,j]=(popt[0])
            mu2_fig2[i,j]=(popt[1])
        except:
            sigma2_fig2[i,j]=(0)
            mu2_fig2[i,j]=(np.nan)
        del DS2
            

#%% Plotting figure 2 - Error vs Epochs and Opts
'''
plt.rc('font', size=12)
fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize=(16.5, 14))
#fig, ax = plt.subplots(nrows = 4, ncols = 2)
for i in range(ax.size):
    i_mod = i//2
    i_dev = i%2
    ax[i_mod,i_dev].title.set_text(str(opts_name[i]))
    ax[i_mod,i_dev].errorbar(np.concatenate([[1],epochs_fig2]), np.concatenate([[popt0[1]],mu1_fig2[i,:]]), 
                             yerr=np.concatenate([[popt0[0]],sigma1_fig2[i,:]]), label='Shift',
                             ls=':',fmt='', color='blue', ecolor='blue', capsize=3)
    ax[i_mod,i_dev].errorbar(np.concatenate([[1],epochs_fig2]), np.concatenate([[popt1[1]],mu2_fig2[i,:]]), 
                             yerr=np.concatenate([[popt1[0]],sigma2_fig2[i,:]]), label='Catmull-Rom Splines',
                             ls=':',fmt='', color='green', ecolor='green', capsize=3)
    
    ax[i_mod,i_dev].set_ylim(1,5e5)
    ax[i_mod,i_dev].set_xscale('log')
    ax[i_mod,i_dev].set_yscale('log')
    ax[i_mod,i_dev].set_xlim(1, epochs_fig2[-1])
    
    if i_mod==0 and i_dev==1: ax[i_mod,i_dev].legend()
    #else: ax[i_mod,i_dev].legend()
    if i_mod==3: ax[i_mod,i_dev].set_xlabel('Iterations')
    else: ax[i_mod,i_dev].set_xticklabels([])
    if i_dev==0: ax[i_mod,i_dev].set_ylabel(r'$\mu$ [nm]')
    else: ax[i_mod,i_dev].set_yticklabels([])

plt.rc('font', size=17)
'''
svfig=['Figure_2a','Figure_2b','Figure_2c','Figure_2d','Figure_2e','Figure_2f','Figure_2g','Figure_2h']
for i in range(len(opts)):
    if i in [4,7]: 
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 2.5))

        ax.set_xlabel('Iterations')
        #ax.set_ylim(1,5e5)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #ax.set_xlim(1, epochs_fig2[-1])
    else:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 2.2))
        '''
        ax.set_ylim(1,5e5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, epochs_fig2[-1])
        ax.set_xticklabels([])
        '''
    #if i in [0,2,4,6,7]: 
    ax.set_ylabel(r'$\mu$ [nm]')
    #else: ax.set_yticklabels([])
    
    #ax.title.set_text(str(opts_name[i]))
    ax.text(x=1.5,y=5e5/2, s=str(opts_name[i]),ha='left', va='top', bbox=dict(boxstyle="square",
                                                           ec=(1., 0.5, 0.5),
                                                           fc=(1., 0.8, 0.8),
                                                           ))
    ax.errorbar(np.concatenate([[1],epochs_fig2]), np.concatenate([[popt0[1]],mu1_fig2[i,:]]), 
                             yerr=np.concatenate([[popt0[0]],sigma1_fig2[i,:]]), label='Shift',
                             ls=':',fmt='', color='blue', ecolor='blue', capsize=3)
    ax.errorbar(np.concatenate([[1],epochs_fig2]), np.concatenate([[popt1[1]],mu2_fig2[i,:]]), 
                             yerr=np.concatenate([[popt1[0]],sigma2_fig2[i,:]]), label='Catmull-Rom Splines',
                             ls=':',fmt='', color='green', ecolor='green', capsize=3)
    
    ax.set_ylim(1,5e5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, epochs_fig2[-1])
    if i==1: ax.legend()
    fig.tight_layout()
    fig.savefig(svfig[i])
    



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#%% fig3 - Error vs Learning-rates 
NN=20
N_it=1
learning_rates1 = 5*np.logspace(-5, 4, NN)
learning_rates2 = np.logspace(-8, 0, NN)

sigma1_fig3,sigma2_fig3=(np.zeros([NN,N_it], dtype=np.float32),np.zeros([NN,N_it], dtype=np.float32))
mu1_fig3,mu2_fig3=(np.zeros([NN,N_it], dtype=np.float32),np.zeros([NN,N_it], dtype=np.float32))

for i in range(NN):
    for j in range(N_it):
        DS1=copy.deepcopy(DS0)
        DS2=copy.deepcopy(DS0shift)
        
        #% DS1 - Shift
        try:
            DS1.ShiftModel=ShiftModel()
            DS1.Train_Model(DS1.ShiftModel, lr=learning_rates1[i], epochs=epochs[0], opt_fn=tf.optimizers.Adagrad)
            DS1.Transform_Model(DS1.ShiftModel)
            popt = ErrorDistribution_r(DS1)
            sigma1_fig3[i,j]=popt[0]
            mu1_fig3[i,j]=popt[1]
        except:
            sigma1_fig3[i,j]=0
            mu1_fig3[i,j]=np.nan
        del DS1
           
        #% DS2  - CR-Splines       
        try:
            ch1_input,ch2_input=DS2.InitializeSplines(gridsize=gridsize, edge_grids=1)
            DS2.SplinesModel=CatmullRomSpline2D(DS2.ControlPoints)
            DS2.Train_Model(DS2.SplinesModel, lr=learning_rates2[i], epochs=epochs[2], opt_fn=tf.optimizers.SGD, 
                             ch1=ch1_input, ch2=ch2_input)  
                
            # transform splines
            DS2.ControlPoints = DS2.SplinesModel.ControlPoints
            DS2.ch2.pos.assign(DS2.InputSplines(
                DS2.Transform_Model(DS2.SplinesModel, ch2=DS2.InputSplines(DS2.ch2.pos)),
                inverse=True))
            DS2.Filter(pair_filter[1])
            popt = ErrorDistribution_r(DS2, GaussianFit=False)
            sigma2_fig3[i,j]=popt[0]
        except:
            sigma2_fig3[i,j]=np.nan
        del DS2


#%% plotting fig3
fig, ax = plt.subplots(figsize=(8.5,7))
ax1 = ax.twinx()
ax.set_xlabel('learning-rate')
ax.set_ylabel(r'$\mu$ [nm]')
ax1.set_ylabel(r'$\sigma$ [nm]')

p1=ax.errorbar(learning_rates1, np.average(mu1_fig3,axis=1), yerr=np.std(mu1_fig3,axis=1), 
               xerr=None, ls=':', fmt='', ecolor='blue', capsize=2, label='Shift')
p3=ax1.errorbar(learning_rates2, np.average(sigma2_fig3,axis=1), yerr= np.std(sigma2_fig3,axis=1),
               xerr=None, ls=':', fmt='', ecolor='green', capsize=2, label='Shift+Catmull-Rom Splines')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_ylim([1,1e3])
#ax1.set_yscale('log')
ax1.set_ylim([1,65])
ax.legend(handles=[p1, p3], loc='upper right')
fig.tight_layout()






#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#%% fig4 - Error vs Gridsize
DS10, DS20=DS0shift.SplitDataset() 

begin=50
end=25000
N=50
Nit=1
gridsizes=np.logspace(np.log10(begin), np.log10(end), N, endpoint=False)
mu1_fig4,mu2_fig4=(np.zeros([len(gridsizes),Nit],dtype=float),np.zeros([len(gridsizes),Nit],dtype=float))
sigma1_fig4,sigma2_fig4=(np.zeros([len(gridsizes),Nit],dtype=float),np.zeros([len(gridsizes),Nit],dtype=float))

for i in range(len(gridsizes)):
    for j in range(Nit):
        try:
            DS1=copy.deepcopy(DS10)
            DS2=copy.deepcopy(DS20)
                
            ch1_input,ch2_input=DS1.InitializeSplines(gridsize=gridsizes[i], edge_grids=1)
            DS1.SplinesModel=CatmullRomSpline2D(DS1.ControlPoints)
            DS1.Train_Model(DS1.SplinesModel, lr=learning_rates[2], epochs=epochs[2], opt_fn=tf.optimizers.SGD, 
                             ch1=ch1_input, ch2=ch2_input)  
                
            # transform splines DS1
            DS1.ControlPoints = DS1.SplinesModel.ControlPoints
            DS1.ch2.pos.assign(DS1.InputSplines(
                DS1.Transform_Model(DS1.SplinesModel, ch2=DS1.InputSplines(DS1.ch2.pos)),
                inverse=True))
            DS1.Filter(pair_filter[1])
            
            # transform splines DS2
            DS2.copy_models(DS1)
            DS2.ControlPoints = DS2.SplinesModel.ControlPoints
            DS2.ch2.pos.assign(DS2.InputSplines(
                DS1.Transform_Model(DS2.SplinesModel, ch2=DS2.InputSplines(DS2.ch2.pos)),
                inverse=True))
            DS2.Filter(pair_filter[1])
            
            sigma1_fig4[i,j] = ErrorDistribution_r(DS1, GaussianFit=False)
            sigma2_fig4[i,j] = ErrorDistribution_r(DS2, GaussianFit=False)
        except:
            sigma1_fig4[i,j]=100
            sigma2_fig4[i,j]=100
        del DS1, DS2
    
    
#%% plotting
xmax=100
#fig, ax1 = plt.subplots(figsize=(16.5,7))
fig, ax1 = plt.subplots(figsize=(8,7))
lns1=ax1.errorbar(gridsizes, np.average(sigma1_fig4,axis=1), yerr=np.std(sigma1_fig4,axis=1), 
                  xerr=None, fmt='',ms=5, color='blue',
                  linestyle=':', label=r'$\sigma_{training}$')
lns2=ax1.errorbar(gridsizes*1.02, np.average(sigma2_fig4,axis=1), yerr=np.std(sigma2_fig4,axis=1),
                  xerr=None, fmt='',ms=5, color='red',
                  linestyle=':', label=r'$\sigma_{testing}$')
lns3=ax1.hlines(DS0.coloc_error, gridsizes[0],gridsizes[-1],linestyle='-.', color='black',label=r'$\sigma_{CRLB}$')

sigma_fig4_avg=np.average((sigma2_fig4),axis=1)
opt_weight=np.min(sigma_fig4_avg)
opt_weight_idx=np.argmin(sigma_fig4_avg)
ax1.vlines(gridsizes[opt_weight_idx], 0, xmax, color='green', linestyle='--', alpha=0.5,
           label=('Optimal Gridsize='+str(round(gridsizes[opt_weight_idx]))+'nm,\n'+r'$\sigma_{training,opt}$='+str(round(sigma_fig4_avg[opt_weight_idx],2))))
ax1.hlines(sigma_fig4_avg[opt_weight_idx],gridsizes[0],gridsizes[-1], linestyle='--', alpha=0.5, color='green')
ax1.legend(loc='upper right')

ax1.set_xlabel('gridsize [nm]')
ax1.set_ylabel(r'$\sigma$ [nm]')
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.set_ylim(1,xmax)
ax1.set_xlim(gridsizes[0],gridsizes[-1])
plt.tight_layout()






#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#%% fig5 - Error vs Density
execute_linked=True
learning_rates = [1e3, .1, 1e-3]
epochs = [100, None, 300]
pair_filter = [None, None, 20]
gridsize=3800

## dataset params
CRS_error=10 # implemented splines error
random_error=3
Num=14000     # number of points
deform_gridsize=1.5*gridsize
loc_error=1.4

Ntimes = [1, 4, 16, 64, 256, 512]
sigma_fig5, N_fig5=(np.zeros([len(Ntimes), 1], dtype=np.float32), np.zeros([len(Ntimes), 1], dtype=np.float32))
for i in range(len(Ntimes)):
    DS1 = dataset_simulation(imgshape=[512, 512], loc_error=loc_error, linked=True,
                             pix_size=159, FrameLinking=False, BatchOptimization=False)
    deform=Affine_Deform()
    DS1.generate_dataset_grid(N=1000*Ntimes[i], deform=deform)
    DS1=SplinesDeform(DS1, gridsize=deform_gridsize, error=CRS_error, random_error=random_error)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    
    DS1.TrainRegistration(execute_linked=execute_linked, learning_rates=learning_rates, 
                      epochs=epochs, pair_filter=pair_filter, gridsize=gridsize)

    if DS2 is not None:
        DS2.copy_models(DS1) ## Copy all mapping parameters
        DS2.ApplyRegistration()
        #DS2.Filter(pair_filter[1])
        sigma_fig5[i,0] = DS2.ErrorDistribution_r(nbins=100, xlim=pair_filter[2],
                                     error=DS1.coloc_error, plot_on=False)
        N_fig5[i,0]=(DS2.ch1.pos.shape[0])
    else: 
        sigma_fig5[i,0] = DS1.ErrorDistribution_r(nbins=100, xlim=pair_filter[2], 
                                             error=DS1.coloc_error, plot_on=False)
        N_fig5[i,0]=(DS1.ch1.pos.shape[0])
    del DS1
    
    
#%% Plotting fig5
def ErrorDist(popt, N, xlim=31, error=None, fig=None):
    ## fit bar plot data using curve_fit
    def func(r, sigma):
        # from Churchman et al 2006
        sigma2=sigma**2
        return r/sigma2*np.exp(-r**2/2/sigma2)
        #return A*(r/sigma2)/(2*np.pi)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)
    
    if fig is None: fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(111)
    x = np.linspace(0, xlim, 1000)
    ## plot how function should look like
    if error is not None:
        sgm=error
        y = func(x, sgm)
        ax.plot(x, y, '--', c='black', label=(r'optimum: $\sigma$='+str(round(sgm,2))+'[nm]'))
        
    for n in range(len(N)):
        y = func(x, popt[n])
        ax.plot(x, y, label=(r'fit: $\sigma$='+str(np.round(popt[n],2))+'[nm], for N='+str(100*round(N[n]/100))))
    

    # Some extra plotting parameters
    ax.set_ylim(0)
    ax.set_xlim([0,xlim])
    ax.set_xlabel('Absolute error [nm]')
    ax.set_ylabel('# of localizations')
    ax.legend(loc='upper right')
    fig.tight_layout()
    
ErrorDist(np.average(sigma_fig5,axis=1), np.average(N_fig5,axis=1), xlim=pair_filter[2], error=DS0.coloc_error)

#%%
print('DONE IN '+str((time.time()-t)//60)+' MINUTES AND '+str(round((time.time()-t)%60,0))+' SECONDS!')