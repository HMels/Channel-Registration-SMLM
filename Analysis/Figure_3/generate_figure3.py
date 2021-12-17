# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:36:49 2021

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
output_path='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_3/Figures/'

from dataset import dataset
from Channel import Channel

plt.rc('font', size=10)
dpi=800
    
    
#%% Load Excel Niekamp
maxDistance=1000
DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
  pix_size=1, loc_error=1.4, mu=0.3,
  linked=False, FrameLinking=True, BatchOptimization=False)
DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
  pix_size=1, loc_error=1.4, mu=0.3,
  linked=False, FrameLinking=True)
DS1.load_dataset_excel()
DS2.load_dataset_excel()
DS1.pix_size=159
DS2.pix_size=DS1.pix_size

## optimization params
execute_linked=True
learning_rate = 1e-3
epochs = 300
pair_filter = [250, 30, 15]
gridsize=6500

## plotting params
nbins=100


DS1.link_dataset(maxDistance=maxDistance)
DS2.link_dataset(maxDistance=maxDistance)
#%% fig1a
plt.close('all')
def generate_2channels(temp, precision=10):
    # Generates the channels as matrix
    print('Generating Channels as matrix...') 
    locs1=temp.ch1.pos.numpy()/precision
    locs2=temp.ch2.pos.numpy()/precision
    temp.precision=precision
    # calculate bounds of the system
    bounds = np.empty([2,2], dtype = float) 
    bounds[0,0] = np.min([ np.min(locs1[:,0]), np.min(locs2[:,0])])
    bounds[0,1] = np.max([ np.max(locs1[:,0]), np.max(locs2[:,0])])
    bounds[1,0] = np.min([ np.min(locs1[:,1]), np.min(locs2[:,1])])
    bounds[1,1] = np.max([ np.max(locs1[:,1]), np.max(locs2[:,1])])
    size_img = np.abs(np.round( (bounds[:,1] - bounds[:,0]) , 0).astype('int')    )        
    axis = np.array([ bounds[1,:], bounds[0,:]]) * temp.precision
    temp.axis = np.reshape(axis, [1,4])[0]
    
    channel = np.zeros([size_img[0]+1, size_img[1]+1], dtype = int)
    for i in range(locs1.shape[0]):
        loc1 = np.round(locs1[i,:],0).astype('int')
        loc2 = np.round(locs2[i,:],0).astype('int')
        loc1 -= np.round(bounds[:,0],0).astype('int')
        loc2 -= np.round(bounds[:,0],0).astype('int')
        channel[loc1[0]-1, loc1[1]-1] = 1
        channel[loc2[0]-1, loc2[1]-1] = -1
    return channel
        
    
def plot_frame(DS1, frame=None):
    temp=copy.deepcopy(DS1)
    if frame is not None:
        framepos1=tf.gather_nd(temp.ch1.pos.numpy(),np.argwhere(temp.ch1.frame==frame))
        framepos2=tf.gather_nd(temp.ch2.pos.numpy(),np.argwhere(temp.ch2.frame==frame))
        del temp.ch1, temp.ch2
        temp.ch1=Channel(framepos1, frame*np.ones(framepos1.shape[0],dtype=float))
        temp.ch2=Channel(framepos2, frame*np.ones(framepos2.shape[0],dtype=float))
    
    #channel=generate_2channels(temp, precision=temp.pix_size*4)
    #channel=np.flipud(channel)
    #fig, ax=temp.plot_1channel(-np.abs(channel), figsize=(1.3,1.3), title='', colormap='gray')
    fig, ax=temp.show_channel(framepos1, color='blue', figsize=(1.3,1.3), ps=2)
    fig, ax=temp.show_channel(framepos2, color='red', fig=fig, ax=ax, ps=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.savefig(output_path+'fig1', transparent=True, dpi=dpi)
    
    
def plot_2channels(DS1):    
    temp=copy.deepcopy(DS1)
    temp.pix_size=100
    
    fig, ax=temp.show_channel(temp.ch1.pos, color='blue', figsize=(1.95,1.95), ps=1, alpha=1, addpatch=False)
    fig, ax=temp.show_channel(temp.ch2.pos, color='red', fig=fig, ax=ax, ps=1, alpha=.7,  addpatch=True)
    ax.add_patch(Rectangle((-4,-4),8,8, ec='red', fc='none'))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.tight_layout()  
    fig.savefig(output_path+'fig2', transparent=True, dpi=dpi)
        
    fig, ax=temp.show_channel(temp.ch1.pos, color='blue', figsize=(1.95,1.95), ps=7, alpha=1, addpatch=False)
    fig, ax=temp.show_channel(temp.ch2.pos, color='red', fig=fig, ax=ax, ps=7, alpha=1,  addpatch=False)
    ax.add_patch(Rectangle((-3.7, -3.7), 2, .1, ec='black', fc='black'))
    ax.text(-3.7, -3.6, r'2$\mu$m', ha='left', va='bottom')
    ax.add_patch(Rectangle((-1.5,-1.5),3,3, ec='blue', fc='none'))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    #fig.tight_layout()
    fig.savefig(output_path+'fig3', transparent=True, dpi=dpi)
    
    fig, ax=temp.show_channel(temp.ch1.pos, color='blue', figsize=(1.95,1.95), ps=14, alpha=1, addpatch=False)
    fig, ax=temp.show_channel(temp.ch2.pos, color='red', fig=fig, ax=ax, ps=14, alpha=1,  addpatch=False)
    ax.add_patch(Rectangle((-1.38, -1.38), .4, .038, ec='black', fc='black'))
    ax.text(-1.38, -1.342, r'400nm', ha='left', va='bottom')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    #fig.tight_layout()
    fig.savefig(output_path+'fig4', transparent=True, dpi=dpi)

   
plot_frame(DS1,1) ## fig 1a
plot_2channels(DS1)


#%% running the model
DS1.AffineLLS()
DS1.Filter(pair_filter[0]) 

#% CatmullRomSplines
if epochs is not None:
    DS1.execute_linked=True #%% Splines can only be optimized by pair-optimization
    DS1.Train_Splines(learning_rate, epochs, gridsize, edge_grids=1)
    DS1.Apply_Splines()
    
DS1.Filter(pair_filter[1])


DS2.copy_models(DS1) ## Copy all mapping parameters
DS2.Apply_Affine(DS1.AffineMat)
if DS2.SplinesModel is not None: DS2.Apply_Splines()
DS2.link_dataset(maxDistance=maxDistance, FrameLinking=True)
DS2.Filter(pair_filter[1])


#%% fig1c
def ErrorDistribution_r(DS1, fig, ax, nbins=30, xlim=31, error=None, mu=.3, fit_data=True):
    if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    dist, avg, r = DS1.ErrorDist(DS1.ch1.pos_all(), DS1.ch2.pos_all())
    dist=dist[np.argwhere(dist<xlim)]
        
    n=ax.hist(dist, range=[0,xlim], alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)#, label='N='+str(DS1.ch1.pos.shape[0]))
    ymax = np.max(n[0])*1.1
    if fit_data: ## fit bar plot data using curve_fit
        def func(r, sigma, mu): # from Churchman et al 2006
            sigma2=sigma**2
            #if mu==0: return r/sigma2*np.exp(-r**2/2/sigma2)
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def load_dataset(path):
    data = pd.read_csv(path)
    pair = np.array(data[['X1','Y1','X2','Y2', 'Distance']])
    average = np.average(pair[:,:2], axis=0)
    pair[:,:2]-=average
    pair[:,2:4]-=average
    return pair


def ErrorDistribution_import(pair, fig, ax, nbins=30, xlim=31, mu=.3, error=None, fit_data=True):
    if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    pos1=DS1.ch1.pos_all()
    dist=pair[:,4]
    
    n = ax.hist(dist, range=[0,xlim], alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)#, label='N='+str(pos1.shape[0]))
    ymax = np.max(n[0]*1.1)
    if fit_data: ## fit bar plot data using curve_fit
        def func(r, sigma, mu): # from Churchman et al 2006
            sigma2=sigma**2
            #if mu==0: return r/sigma2*np.exp(-r**2/2/sigma2)
            return (r/sigma2)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)

        N = pos1.shape[0] * ( n[1][1]-n[1][0] )
        xn=(n[1][:-1]+n[1][1:])/2 
        popt, pcov = curve_fit(func, xn, n[0]/N, p0=[np.std(xn), np.average(xn)])
        x = np.linspace(0, xlim, 1000)
        y = func(x, *popt)*N
        ax.plot(x, y, c='g',label=(r'$\sigma$='+str(round(popt[0],2))+'nm\n$\mu$='+str(round(popt[1],2))+'nm'))
    
        if error is not None: ## plot how function should look like
            sgm=np.sqrt(2)*error
            y = func(x, sgm, mu)*N
            ax.plot(x, y, c='b',label=(r'$\sigma$='+str(round(sgm,2))+'nm, $\mu$='+str(round(mu,2))+'nm'))
            if np.max(y)>ymax: ymax=np.max(y)*1.1

    # Some extra plotting parameters
    ax.set_ylim([0,ymax])
    ax.set_xlim([0,xlim])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


fig=plt.figure(figsize=(7.2,2.56))
ax1 = fig.add_subplot(221)
fig, ax1=ErrorDistribution_r(DS1, fig, ax1, nbins=nbins, xlim=pair_filter[2], error=DS1.loc_error, mu=DS1.mu, fit_data=True)

ax2 = fig.add_subplot(222)
fig, ax2=ErrorDistribution_r(DS2, fig, ax2, nbins=nbins, xlim=pair_filter[2], error=DS2.loc_error, mu=DS2.mu, fit_data=True)

ax3 = fig.add_subplot(223)
DS3 = load_dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration_After_Mapping/Set1/set1_beads_locs.csv')
fig, ax3=ErrorDistribution_import(DS3, fig, ax3, nbins=nbins, xlim=pair_filter[2], error=DS1.loc_error, mu=DS1.mu, fit_data=True)

ax4 = fig.add_subplot(224)
DS4 = load_dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration_After_Mapping/Set2/set2_beads_locs.csv')
fig, ax4=ErrorDistribution_import(DS4, fig, ax4, nbins=nbins, xlim=pair_filter[2], error=DS1.loc_error, mu=DS1.mu, fit_data=True)

# fixing the axis
ylim=np.max([ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1],ax4.get_ylim()[1]])
ax1.set_ylim([0,ylim])
ax2.set_ylim([0,ylim])
ax3.set_ylim([0,ylim])
ax4.set_ylim([0,ylim])
ax4.set_xticks([0,int(pair_filter[2]/3), int(pair_filter[2]*2/3),pair_filter[2]])
ax3.set_xticks([0,int(pair_filter[2]/3), int(pair_filter[2]*2/3),pair_filter[2]])
fig.text(0.5,0.04, 'absolute error [nm]', ha="center", va="center")
#fig.text(0.05,0.5, "Pair distance distribution", ha="center", va="center", rotation=90)
fig.tight_layout(h_pad=0)
height1=ylim*.7
height2=ylim*.5
ax1.text(x=pair_filter[2]*.97,y=height1, s=r'$CRsCR$ $estimation$', color='black',ha='right', va='top')
ax2.text(x=pair_filter[2]*.97,y=height1, s=r'$CRsCR$ $testing$', color='black',ha='right', va='top')
ax3.text(x=pair_filter[2]*.97,y=height1, s=r'$Niekamp$ $estimation$', color='black',ha='right', va='top')
ax4.text(x=pair_filter[2]*.97,y=height1, s=r'$Niekamp$ $testing$', color='black',ha='right', va='top')
ax1.text(x=pair_filter[2]*.97,y=height2, s=ax1.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax2.text(x=pair_filter[2]*.97,y=height2, s=ax2.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax3.text(x=pair_filter[2]*.97,y=height2, s=ax3.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax4.text(x=pair_filter[2]*.97,y=height2, s=ax4.get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig.savefig(output_path+'fig5', transparent=True, dpi=dpi)

lbls = []
for ax in fig.axes:
    Line, Label = ax.get_legend_handles_labels()
    lbls.extend(Label)
#lblsplus=['CRsCR estimation: ', 'CRsCR testing: ', 'Niekamp estimation: ', 
#          'Niekamp testing: ', 'Lower Bound: ']
txt=''
for i in range(len(lbls)):
    #txt+=lblsplus[i]+lbls[i]+'\n'
    txt+=lbls[i]+'\n'
file2write=open(output_path+'fig3.txt','w')
file2write.write(txt)
file2write.close()


#%% fig1gh
def ErrorDistribution_xy(DS1, fig, ax, nbins=30, xlim=31, error=None, mu=None, fit_data=True, comparison=None):
        #if not DS1.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        def func(r, mu, sigma):
            return np.exp(-(r - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2*np.pi)*sigma)
        
        def fit_curve(distx, ax=None):
            if  ax is None:
                nx=np.histogram(distx, bins=nbins)
            else:
                nx = ax.hist(distx, range=[-xlim,xlim],alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
            Nx = pos1.shape[0] * ( nx[1][1]-nx[1][0] )
            xn=(nx[1][:-1]+nx[1][1:])/2
            poptx, pcovx = curve_fit(func, xn, nx[0]/Nx, p0=[np.average(distx), np.std(distx)])
            return poptx, Nx, nx
            
        if mu is None: mu=0
        pos1=DS1.ch1.pos.numpy()
        pos2=DS1.ch2.pos.numpy()                
        distx=pos1[:,0]-pos2[:,0]
        disty=pos1[:,1]-pos2[:,1]
        mask=np.where(distx<xlim,True, False)*np.where(disty<xlim,True,False)
        distx=distx[mask]
        disty=disty[mask]
        poptx, Nx, nx=fit_curve(distx, ax=ax[0])
        popty, Ny, ny=fit_curve(disty, ax=ax[1])
        
        if comparison is not None:
            comp1=comparison[:,:2]
            comp2=comparison[:,2:4]
            compx=comp1[:,0]-comp2[:,0]
            compy=comp1[:,1]-comp2[:,1]
            compx=compx[mask]
            compy=compy[mask]
            mask=np.where(compx<xlim,True, False)*np.where(compx<xlim,True,False)
            poptxc, Nxc, nxc=fit_curve(compx)
            poptyc, Nyc, nyc=fit_curve(compy)
            
        if fit_data: 
            x = np.linspace(-xlim, xlim, 1000)
            yx = func(x, *poptx)*Nx
            yy = func(x, *popty)*Ny
            ax[0].plot(x, yx, c='g',label=('$CRsCR$ \n'+r'$\sigma$='+str(round(poptx[1],2))+'nm\n$\mu$='+str(round(poptx[0],2))+'nm'))
            ax[1].plot(x, yy, c='g',label=('$CRsCR$ \n'+r'$\sigma$='+str(round(popty[1],2))+'nm\n$\mu$='+str(round(popty[0],2))+'nm'))
            ymax = np.max([np.max(nx[0]),np.max(ny[0]), np.max(yx), np.max(yy)])*1.1
            '''
            if comparison is not None:
                yxc = func(x, *poptxc)*Nxc
                yyc = func(x, *poptyc)*Nyc
                ax[0].plot(x, yxc, c='red',label=(r'$\sigma$='+str(round(poptxc[1],2))+'nm\n$\mu$='+str(round(poptxc[0],2))+'nm'))
                ax[1].plot(x, yyc, c='red',label=(r'$\sigma$='+str(round(poptyc[1],2))+'nm\n$\mu$='+str(round(poptyc[0],2))+'nm'))
            '''                          
            
            ## plot how function should look like
            if error is not None:
                sgm=np.sqrt(2)*error+mu
                opt_yx = func(x, 0, sgm)*Nx
                opt_yy = func(x, 0, sgm)*Ny
                ax[0].plot(x, opt_yx, c='b',label=(r'lower bound: $\mu$='+str(round(mu,2))+', $\sigma$='+str(round(sgm,2))+'nm'))
                ax[1].plot(x, opt_yy, c='b')
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
        if comparison is None:
            return fig, ax
        else:
            return fig, ax, poptxc, poptyc


##fig4
fig, ax = plt.subplots(1,2,figsize=(4.88*1.6,1.2))
fig, ax, poptxc, poptyc= ErrorDistribution_xy(DS1, fig, ax, nbins=nbins, xlim=pair_filter[2], 
                                  error=DS2.loc_error, comparison=DS3)
fig.axes[0].set_xlabel('')
fig.axes[1].set_xlabel('')
fig.axes[0].set_xticks([])
fig.axes[1].set_xticks([])
ax[0].text(x=pair_filter[2]*.9,y=ylim, s=ax[0].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax[1].text(x=pair_filter[2]*.9,y=ylim, s=ax[1].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax[0].text(x=-pair_filter[2]*.9,y=ylim, s=('$Niekamp$ \n'+r'$\sigma$='+str(round(poptxc[1],2))+'nm\n$\mu$='+str(round(poptxc[0],2))+'nm'), 
           color='black',ha='left', va='top')
ax[1].text(x=-pair_filter[2]*.9,y=ylim, s=('$Niekamp$ \n'+r'$\sigma$='+str(round(poptyc[1],2))+'nm\n$\mu$='+str(round(poptyc[0],2))+'nm'), 
           color='black',ha='left', va='top')
fig.text(.5, .1, '$estimation$', color='black',ha='center', va='center')
fig.tight_layout()
fig.savefig(output_path+'fig6a', transparent=True, dpi=dpi)

fig, ax = plt.subplots(1,2,figsize=(4.88*1.6,1.6))
fig, ax, poptxc, poptyc= ErrorDistribution_xy(DS2, fig, ax, nbins=nbins, xlim=pair_filter[2],
                                  error=DS2.loc_error, comparison=DS4)
ax[0].text(x=pair_filter[2]*.9,y=ylim, s=ax[0].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax[1].text(x=pair_filter[2]*.9,y=ylim, s=ax[1].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax[0].text(x=-pair_filter[2]*.9,y=ylim, s=('$Niekamp$ \n'+r'$\sigma$='+str(round(poptxc[1],2))+'nm\n$\mu$='+str(round(poptxc[0],2))+'nm'), 
           color='black',ha='left', va='top')
ax[1].text(x=-pair_filter[2]*.9,y=ylim, s=('$Niekamp$ \n'+r'$\sigma$='+str(round(poptyc[1],2))+'nm\n$\mu$='+str(round(poptyc[0],2))+'nm'), 
           color='black',ha='left', va='top')
fig.text(.5, .26, '$testing$', color='black',ha='center', va='center')
fig.tight_layout()
fig.savefig(output_path+'fig6b', transparent=True, dpi=dpi)


'''
fig, ax = plt.subplots(1,2,figsize=(4.88*1.6,1.95))
fig, ax, poptx1, popty1= ErrorDistribution_xy(DS3, fig, ax, nbins=nbins, xlim=pair_filter[2], error=DS2.loc_error)
ax[0].text(x=pair_filter[2]*.9,y=ylim*.9, s='Niekamp estimation\n'+ax[0].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax[1].text(x=pair_filter[2]*.9,y=ylim*.9, s='Niekamp estimation\n'+ax[1].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig.tight_layout()
#fig.savefig(output_path+'fig6c', transparent=True, dpi=dpi)

fig, ax = plt.subplots(1,2,figsize=(4.88*1.6,1.95))
fig, ax, poptx2, popty2= ErrorDistribution_xy(DS4, fig, ax, nbins=nbins, xlim=pair_filter[2], error=DS2.loc_error)
ax[0].text(x=pair_filter[2]*.9,y=ylim*.9, s='Niekamp testing\n'+ax[0].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
ax[1].text(x=pair_filter[2]*.9,y=ylim*.9, s='Niekamp testing\n'+ax[1].get_legend_handles_labels()[1][0], color='black',ha='right', va='top')
fig.tight_layout()
#fig.savefig(output_path+'fig6d', transparent=True, dpi=dpi)
'''


(lbls,txt)=([],'')
for ax1 in fig.axes:
    Line, Label = ax1.get_legend_handles_labels()
    lbls.extend(Label)
for lbl in lbls:
    txt+=lbl+'\n'
file2write=open(output_path+'fig6.txt','w')
file2write.write(txt)
file2write.close()


#%% fig5
fig,axFOV=DS1.ErrorFOV(None, figsize=(4.148,1.93),placement='right', center=[4.8,4.8])
fig.savefig(output_path+'fig7', transparent=True, dpi=dpi)
fig,axFOV=DS2.ErrorFOV(None, figsize=(3.66,1.93),placement='right', center=[4.8,4.8], colorbar=False)
fig.savefig(output_path+'fig8', transparent=True, dpi=dpi)