# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:18:00 2021

@author: Mels
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:03:46 2021

@author: Mels
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as scpspc
import matplotlib as mpl
import tensorflow as tf

class Plot:
    def __init__(self):
        pass

    def ErrorDist(self, pos1, pos2):
    # Generates the error, average and radius
        dist = np.sqrt( np.sum( ( pos1 - pos2 )**2, axis = 1) )
        return dist, np.average(dist), np.sqrt(np.sum(pos1**2,1)) 
    
    
    #%% Plotting the error
    def ErrorPlot(self, nbins=30):
        ## Coupling Datasets if not done already
        if not self.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        pos1_original=self.ch1.pos_all()
        pos2_original=self.ch20linked.pos_all()
        pos1=self.ch1.pos_all()
        pos2=self.ch2.pos_all()
        
        # Calculating the error
        dist1, avg1, r1 = self.ErrorDist(pos1, pos2)
        if self.ch20linked.pos_all() is not None: 
            dist2, avg2, r2 = self.ErrorDist(pos1_original, pos2_original)
        
        
        ## Plotting
        if self.ch20linked.pos_all() is not None: fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(2,2)
        else: fig, (ax1, ax2) = plt.subplots(2)
          
        # plotting the histogram
        n1 = ax1.hist(dist1, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        if self.ch20linked.pos_all() is not None:
            n1 = ax3.hist(dist1, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
            n2 = ax3.hist(dist2, label='Original', alpha=.8, edgecolor='red', color='tab:blue', bins=nbins)
        else:
            n2=[0]
        ymax = np.max([np.max(n1[0]), np.max(n2[0])])*1.1
            
        # plotting the FOV
        ax2.plot(r1, dist1, 'r.', alpha=.4, label='Mapped error')
        if self.ch20linked.pos_all() is not None:
            ax4.plot(r1, dist1, 'r.', alpha=.4, label='Mapped error')
            ax4.plot(r2, dist2, 'b.', alpha=.4, label='Original error') 
        else:
            r2=np.array(0)
        xmax= np.max((np.max(r1),np.max(r2)))*1.1
        
        # Plotting the averages as vlines
        ax1.vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg1,2))))
        if self.ch20linked.pos_all() is not None:
            ax3.vlines(avg2, color='purple', ymin=0, ymax=ymax, label=('avg original = '+str(round(avg2,2))))
            ax3.vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg1,2))))
          
        # Plotting the averages as hlines
        ax2.hlines(avg1, color='green', xmin=0, xmax=xmax, label=('average mapped = '+str(round(avg1,2))))
        if self.ch20linked.pos_all() is not None:
            ax4.hlines(avg2, color='purple', xmin=0, xmax=xmax, label=('average original = '+str(round(avg2,2))))
            ax4.hlines(avg1, color='green', xmin=0, xmax=xmax, label=('average mapped = '+str(round(avg1,2))))
        
        
        # Some extra plotting parameters
        ax1.set_title('Zoomed in on Mapping Error')
        ax1.set_ylim([0,ymax])
        ax1.set_xlim(0)
        ax1.set_xlabel('Absolute error [nm]')
        ax1.set_ylabel('# of localizations')
        ax1.legend()
        
        ax2.set_title('Zoomed in on Mapping Error')
        ax2.set_ylim(0)
        ax2.set_xlim([0,xmax])
        ax2.set_xlabel('FOV [nm]')
        ax2.set_ylabel('Absolute Error')
        ax2.legend()
        
        if self.ch20linked.pos_all() is not None:
            ax3.set_title('Comparisson')
            ax3.set_ylim([0,ymax])
            ax3.set_xlim(0)
            ax3.set_xlabel('Absolute error [nm]')
            ax3.set_ylabel('# of localizations')
            ax3.legend()
        
            ax4.set_title('Comparisson')
            ax4.set_ylim(0)
            ax4.set_xlim([0,xmax])
            ax4.set_xlabel('FOV [nm]')
            ax4.set_ylabel('Absolute Error')
            ax4.legend()
          
        fig.tight_layout()
        fig.show()
        if self.ch20linked.pos_all() is not None: 
            print('The original model had an average absolute error of',avg2,'nm\nThe mapped model has an average absolute error of',avg1,'nm')
            return avg1, avg2, fig, (ax3, ax1, ax4, ax2)
        else: 
            print('The mapped model has an average absolute error of',avg1,'nm')
            return avg1, fig, (ax1, ax2)


    def ErrorDistribution(self, nbins=30):
    # just plots the error distribution after mapping
        if not self.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        pos1=self.ch1.pos_all()
        pos2=self.ch2.pos_all()
        dist1, avg1, r1 = self.ErrorDist(pos1, pos2)
        
        plt.figure()
        n1 = plt.hist(dist1, label=('Mapped Error = '+str(round(avg1,2))+'nm'), alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        ymax = np.max(n1[0]*1.1)
        #plt.title('Zoomed in on Mapping Error')
        plt.ylim([0,ymax])
        plt.xlim(0)
        plt.xlabel('error [nm]')
        plt.ylabel('# of localizations')
        plt.legend()
        plt.tight_layout()
        
        
    def ErrorDistribution_xy(self, nbins=30, xlim=31, error=None):
        if not self.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        pos1=self.ch1.pos_all()
        pos2=self.ch2.pos_all()
            
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        distx=pos1[:,0]-pos2[:,0]
        disty=pos1[:,1]-pos2[:,1]
        mask=np.where(distx<xlim,True, False)*np.where(disty<xlim,True,False)
        distx=distx[mask]
        disty=disty[mask]
        nx = ax[0].hist(distx, label='data',alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        ny = ax[1].hist(disty, label='data',alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        
         ## fit bar plot data using curve_fit
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
        ax[0].plot(x, yx, c='g',label=(r'fit: $\mu$='+str(round(poptx[0],2))+', $\sigma$='+str(round(poptx[1],2))+'[nm]'))
        ax[1].plot(x, yy, c='g',label=(r'fit: $\mu$='+str(round(popty[0],2))+', $\sigma$='+str(round(popty[1],2))+'[nm]'))
        ymax = np.max([np.max(nx[0]),np.max(ny[0]), np.max(yx), np.max(yy)])*1.1
        
        ## plot how function should look like
        if error is not None:
            sgm=np.sqrt(2)*error
            opt_yx = func(x, 0, sgm)*Nx
            opt_yy = func(x, 0, sgm)*Ny
            ax[0].plot(x, opt_yx, c='b',label=(r'optimum: $\sigma$='+str(round(sgm,2))+'[nm] ($\sqrt{2}$ loc-error)'))
            ax[1].plot(x, opt_yy, c='b',label=(r'optimum: $\sigma$='+str(round(sgm,2))+'[nm] ($\sqrt{2}$ loc-error)'))
            if np.max([np.max(opt_yx),np.max(opt_yy)])>ymax: ymax=np.max([np.max(opt_yx),np.max(opt_yy)])*1.1

        ax[0].set_ylim([0,ymax])
        ax[0].set_xlim(-xlim,xlim)
        ax[0].set_xlabel('x-error [nm]')
        ax[0].set_ylabel('# of localizations')
        ax[0].legend()
        
        ax[1].set_ylim([0,ymax])
        ax[1].set_xlim(-xlim,xlim)
        ax[1].set_xlabel('y-error [nm]')
        ax[1].set_ylabel('# of localizations')
        ax[1].legend()
        fig.tight_layout()
        
        return poptx, popty
        
        
    def ErrorDistribution_r(self, nbins=30, xlim=31, error=None):
        if not self.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        pos1=self.ch1.pos_all()
        pos2=self.ch2.pos_all()
            
        # Calculating the error
        dist, avg, r = self.ErrorDist(pos1, pos2)
        dist=dist[np.argwhere(dist<xlim)]
        
        # plotting the histogram
        plt.figure()
        n=plt.hist(dist, label='N='+str(pos1.shape[0]), alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        #plt.axvline(x=avg, label='average='+str(round(avg,2))+'[nm]')
        ymax = np.max(n[0])*1.1
        
        
        ## fit bar plot data using curve_fit
        def func(r, sigma):
            # from Churchman et al 2006
            sigma2=sigma**2
            return r/sigma2*np.exp(-r**2/2/sigma2)
            #return A*(r/sigma2)/(2*np.pi)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)
        
        N = pos1.shape[0] * ( n[1][1]-n[1][0] )
        xn=(n[1][:-1]+n[1][1:])/2
        popt, pcov = curve_fit(func, xn, n[0]/N, p0=np.std(xn))
        x = np.linspace(0, xlim, 1000)
        y = func(x, *popt)*N
        plt.plot(x, y, c='g',label=(r'fit: $\sigma$='+str(round(popt[0],2))+'[nm] ($\mu$ is kept at 0 [nm])'))
        
        ## plot how function should look like
        if error is not None:
            sgm=np.sqrt(2)*error
            y = func(x, sgm)*N
            plt.plot(x, y, c='b',label=(r'optimum: $\sigma$='+str(round(sgm,2))+'[nm] ($\sqrt{2}$ loc-error)'))
            if np.max(y)>ymax: ymax=np.max(y)*1.1

        # Some extra plotting parameters
        plt.ylim([0,ymax])
        plt.xlim([0,xlim])
        plt.xlabel('Absolute error [nm]')
        plt.ylabel('# of localizations')
        plt.legend()
        plt.tight_layout()
        return popt
        

    #%% plotting the error in a [x1, x2] plot like in the paper        
    def ErrorFOV(self, other=None, maxDist=30, ps=5, cmap='seismic'):
        ## Coupling Dataset1 if not done already
        if not self.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
        dist = self.ch1.pos_all()-self.ch2.pos_all()
        if dist.shape==(0,): raise ValueError('No neighbours found for channel 1')
            
        ## Coupling Dataset2 if not done already
        if other is not None:
            if not self.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
            dist1 = other.ch1.pos-other.ch2.pos
            if dist1.shape==(0,): raise ValueError('No neighbours found for channel 2')
            
            vmin=np.min((np.min(dist[:,0]),np.min(dist1[:,0]),np.min(dist[:,1]),np.min(dist1[:,1])))
            vmax=np.max((np.max(dist[:,0]),np.max(dist1[:,0]),np.max(dist[:,1]),np.max(dist1[:,1])))
            
            
            fig, ax = plt.subplots(2,2)
            ax[0][0].scatter(self.ch1.pos_all()[:,0]/1000, self.ch1.pos_all()[:,1]/1000, s=ps, c=dist[:,0], cmap=cmap, vmin=vmin, vmax=vmax)
            #ax[0][0].set_xlabel('x-position [\u03bcm]')
            ax[0][0].set_ylabel('Set 1 Fiducials\ny-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-error [nm]', ax=ax[0][0])
            ax[0][0].set_aspect('equal', 'box')
            
            ax[0][1].scatter(self.ch1.pos_all()[:,0]/1000, self.ch1.pos_all()[:,1]/1000, s=ps, c=dist[:,1], cmap=cmap, vmin=vmin, vmax=vmax)
            #ax[0][1].set_xlabel('x-position [\u03bcm]')
            #ax[0][1].set_ylabel('y-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-error [nm]', ax=ax[0][1])
            ax[0][1].set_aspect('equal', 'box')
        
            ax[1][0].scatter(other.ch1.pos[:,0]/1000, other.ch1.pos[:,1]/1000, s=ps, c=dist1[:,0], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[1][0].set_xlabel('x-position [\u03bcm]')
            ax[1][0].set_ylabel('Set 2 Fiducials\ny-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-error [nm]', ax=ax[1][0])
            ax[1][0].set_aspect('equal', 'box')
            
            ax[1][1].scatter(other.ch1.pos[:,0]/1000, other.ch1.pos[:,1]/1000, s=ps, c=dist1[:,1], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[1][1].set_xlabel('x-position [\u03bcm]')
            #ax[1][1].set_ylabel('y-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-error [nm]', ax=ax[1][1])
            ax[1][1].set_aspect('equal', 'box')
            fig.tight_layout()
            
        else:
            vmin=np.min((np.min(dist[:,0]),np.min(dist[:,1])))
            vmax=np.max((np.max(dist[:,0]),np.max(dist[:,1])))
            fig, ax = plt.subplots(1,2)
            ax[0].scatter(self.ch1.pos_all()[:,0]/1000, self.ch1.pos_all()[:,1]/1000, s=ps, c=dist[:,0], cmap=cmap)
            ax[0].set_xlabel('x-position [\u03bcm]')
            ax[0].set_ylabel('Batch 1\ny-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-error [nm]', ax=ax[0])
            ax[0].set_aspect('equal', 'box')
            
            ax[1].scatter(self.ch1.pos_all()[:,0]/1000, self.ch1.pos_all()[:,1]/1000, s=ps, c=dist[:,1], cmap=cmap)
            ax[1].set_xlabel('x-position [\u03bcm]')
            ax[1].set_ylabel('y-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-error [nm]', ax=ax[1])
            ax[1].set_aspect('equal', 'box')
            fig.tight_layout()
        
    
        
    #%% Channel to matrix fn
    def isin_domain(self, pos):
    # checks if pos is within bounds
        return ( pos[0] > self.bounds[0,0] and pos[1] > self.bounds[1,0] and 
                pos[0] < self.bounds[0,1] and pos[1] < self.bounds[1,1] )
    
    
    def generate_channel(self, precision=10, heatmap=False):
    # Generates the channels as matrix
        print('Generating Channels as matrix...')       
    
        # normalizing system
        if self.ch10.pos_all() is not None: locs1 = self.ch10.pos_all()  / precision
        else:locs1 = self.ch1.pos_all()  / precision
        locs2 = self.ch2.pos_all()  / precision
        if self.ch20.pos_all() is not None: locs2_original = self.ch20.pos_all()  / precision
        else: locs2_original=locs2
        
        # calculate bounds of the system
        self.precision=precision
        self.bounds = np.empty([2,2], dtype = float) 
        self.bounds[0,0] = np.min([ np.min(locs1[:,0]), np.min(locs2[:,0]), np.min(locs2_original[:,0]) ])
        self.bounds[0,1] = np.max([ np.max(locs1[:,0]), np.max(locs2[:,0]), np.max(locs2_original[:,0]) ])
        self.bounds[1,0] = np.min([ np.min(locs1[:,1]), np.min(locs2[:,1]), np.min(locs2_original[:,1]) ])
        self.bounds[1,1] = np.max([ np.max(locs1[:,1]), np.max(locs2[:,1]), np.max(locs2_original[:,1]) ])
        self.size_img = np.abs(np.round( (self.bounds[:,1] - self.bounds[:,0]) , 0).astype('int')    )        
        self.axis = np.array([ self.bounds[1,:], self.bounds[0,:]]) * self.precision
        self.axis = np.reshape(self.axis, [1,4])[0]
        
        # generating the matrices to be plotted
        self.channel1 = self.generate_matrix(locs1, heatmap)
        self.channel2 = self.generate_matrix(locs2, heatmap)
        if self.ch20.pos_all() is not None: self.channel2_original = self.generate_matrix(locs2_original, heatmap)
        
        
    def generate_matrix(self, locs, heatmap=False):
    # takes the localizations and puts them in a channel
        channel = np.zeros([self.size_img[0]+1, self.size_img[1]+1], dtype = int)
        for i in range(locs.shape[0]):
            loc = np.round(locs[i,:],0).astype('int')
            if self.isin_domain(loc):
                loc -= np.round(self.bounds[:,0],0).astype('int') # place the zero point on the left
                if heatmap: channel[loc[0], loc[1]] += 1
                else: channel[loc[0], loc[1]] = 1
        return channel
    
    
    #%% Plotting Channels
    def plot_channel(self):
        print('Plotting...')
        rotate=self.channel1.shape[1]>self.channel1.shape[0]
        if rotate:
            self.channel1=np.rot90(self.channel1)
            self.channel2=np.rot90(self.channel2)
            self.channel2_original=np.rot90(self.channel2_original)
            self.axis=np.concatenate((self.axis[2:],self.axis[:2]))
            label=['x1','x2']
        else:
            label=['x2', 'x1']
            
        # plotting all channels
        plt.figure()
        if self.ch20.pos_all() is not None: plt.subplot(131)
        else: plt.subplot(121)
        plt.imshow(self.channel1, extent = self.axis)
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.title('original channel 1')
        plt.tight_layout()
        
        if self.ch20.pos_all() is not None: plt.subplot(132)
        else: plt.subplot(122)
        plt.imshow(self.channel2, extent = self.axis)
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.title('mapped channel 2')
        plt.tight_layout()
        
        if self.ch20.pos_all() is not None: 
            plt.subplot(133)
            plt.imshow(self.channel2_original, extent = self.axis)
            plt.xlabel(label[0])
            plt.ylabel(label[1])
            plt.title('original channel 2')
            plt.tight_layout()
        
        
    def plot_1channel(self, channel1=None):
        if channel1 is None: channel1=self.channel1
        if channel1.shape[1]>channel1.shape[0]: channel1=np.rot90(channel1)
        # plotting all channels
        plt.figure()
        plt.imshow(channel1, extent = self.axis)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Single Channel view')
        plt.tight_layout()
        
        
    #%% Plotting the Grid
    def PlotSplineGrid(self, gridsize=None, edge_grids=None, ch1=None, ch2=None, ch20=None, locs_markersize=25,
                        CP_markersize=20, d_grid=.1, Ngrids=4, plotarrows=True, plotmap=False): 
        '''
        Plots the grid and the shape of the grid in between the Control Points
    
        Parameters
        ----------
        ch1 , ch2 , ch20 : Nx2 tf.float32 tensor
            The tensor containing the localizations.
        d_grid : float, optional
            The precission of the grid we want to plot in between the
            ControlPoints. The default is .1.
        lines_per_CP : int, optional
            The number of lines we want to plot in between the grids. 
            Works best if even. The default is 1.
        locs_markersize : float, optional
            The size of the markers of the localizations. The default is 10.
        CP_markersize : float, optional
            The size of the markers of the Controlpoints. The default is 8.
            
        Returns
        -------
        None.
    
        '''
        print('Plotting the Spline Grid...')
        if ch1 is None:
            ch1=self.ch1.pos
            ch2=self.ch2.pos
            ch20=self.ch20.pos
        if gridsize is None: gridsize=self.gridsize
        if edge_grids is None: edge_grids=self.edge_grids
        if self.SplinesModel is None:
            x1_min = np.min([np.min(tf.reduce_min(tf.floor(ch1[:,0]))),
                                  np.min(tf.reduce_min(tf.floor(ch2[:,0])))])/gridsize
            x2_min = np.min([np.min(tf.reduce_min(tf.floor(ch1[:,1]))),
                                  np.min(tf.reduce_min(tf.floor(ch2[:,1])))])/gridsize
            x1_max = np.max([np.max(tf.reduce_max(tf.floor(ch1[:,0]))),
                                  np.max(tf.reduce_max(tf.floor(ch2[:,0])))])/gridsize
            x2_max = np.max([np.max(tf.reduce_max(tf.floor(ch1[:,1]))),
                                  np.max(tf.reduce_max(tf.floor(ch2[:,1])))])/gridsize
            x1_grid = tf.range(0, tf.math.ceil(x1_max)-x1_min+edge_grids+1+d_grid, dtype=tf.float32)
            x2_grid = tf.range(0, tf.math.ceil(x2_max)-x2_min+edge_grids+1+d_grid, dtype=tf.float32)
            ControlPoints = tf.stack(tf.meshgrid(x1_grid, x2_grid), axis=-1)
        else:
            x1_min = self.x1_min
            x2_min = self.x2_min
            x1_max = self.x1_max
            x2_max = self.x2_max
            ControlPoints = self.ControlPoints
        
        ## The original points
        ch1 = tf.Variable( tf.stack([
            (ch1[:,0] - np.min(ch20[:,0]) ) + edge_grids*gridsize,
            (ch1[:,1] - np.min(ch20[:,1])) + edge_grids*gridsize
            ], axis=-1), dtype=tf.float32, trainable=False)
        ch2 = tf.Variable( tf.stack([
            (ch2[:,0] - np.min(ch20[:,0]) ) + edge_grids*gridsize,
            (ch2[:,1] - np.min(ch20[:,1]) ) + edge_grids*gridsize
            ], axis=-1), dtype=tf.float32, trainable=False)
        ch20 = tf.Variable( tf.stack([
            (ch20[:,0] - np.min(ch20[:,0]) ) + edge_grids*gridsize,
            (ch20[:,1] - np.min(ch20[:,1]) ) + edge_grids*gridsize
            ], axis=-1), dtype=tf.float32, trainable=False)
        
        # plotting the localizations
        plt.figure()
        plt.scatter(ch20[:,0],ch20[:,1], c='green', marker='.', s=locs_markersize, label='Original')
        plt.scatter(ch1[:,0],ch1[:,1], c='red', marker='.', s=locs_markersize, label='Target')
        if plotarrows:
            for i in range(ch1.shape[0]):
                plt.arrow(ch20[i,0],ch20[i,1], ch2[i,0]-ch20[i,0], ch2[i,1]-ch20[i,1], width=.02, 
                          length_includes_head=True, facecolor='red', edgecolor='red')
        if plotmap: 
            plt.scatter(ch2[:,0],ch2[:,1], c='blue', marker='.', s=locs_markersize, label='Mapped')
        
        # plotting the ControlPoints
        plt.scatter(ControlPoints[:,:,0]*gridsize, ControlPoints[:,:,1]*gridsize,
                    c='b', marker='d', s=CP_markersize, label='ControlPoints')
        
        ## Horizontal Grid
        x1_grid = tf.range(0, tf.math.ceil(x1_max)-x1_min+edge_grids+1+d_grid, delta=d_grid, dtype=tf.float32)
        x2_grid = tf.range(0, tf.math.ceil(x2_max)-x2_min+edge_grids+1+d_grid, delta=1/Ngrids, dtype=tf.float32)
        Grid = tf.reshape(tf.stack(tf.meshgrid(x1_grid, x2_grid), axis=-1) , (-1,2)) 
        if self.SplinesModel is not None: Grid = self.SplinesModel( Grid ) * gridsize
        else: Grid = Grid*gridsize
        (nn, i,j)=(x1_grid.shape[0],0,0)
        while i<Grid.shape[0]:
            if j%Ngrids==0:
                plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='b')
            else:
                plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='c')
            i+=nn
            j+=1

        ## Vertical Grid
        x1_grid = tf.range(0, tf.math.ceil(x1_max)-x1_min+edge_grids+1+d_grid, delta=1/Ngrids, dtype=tf.float32)
        x2_grid = tf.range(0, tf.math.ceil(x2_max)-x2_min+edge_grids+1+d_grid, delta=d_grid, dtype=tf.float32)
        Grid = tf.gather(tf.reshape(tf.stack(tf.meshgrid(x2_grid, x1_grid), axis=-1) , (-1,2)), [1,0], axis=1)
        if self.SplinesModel is not None: Grid = self.SplinesModel( Grid ) * gridsize
        else: Grid = Grid*gridsize
        (nn, i,j)=(x2_grid.shape[0],0,0)
        while i<Grid.shape[0]:
            if j%Ngrids==0:
                plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='b')
            else:
                plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='c')
            i+=nn
            j+=1
        
        plt.legend()
        plt.tight_layout()
            

    def PlotGridMapping(self, model, gridsize, edge_grids=None, ch1=None, ch2=None, ch20=None, 
                            locs_markersize=25, CP_markersize=25, d_grid=.1, Ngrids=4, plotarrows=True, plotmap=False): 
            '''
            Plots the grid and the shape of the grid in between the Control Points
        
            Parameters
            ----------
            ch1 , ch2 , ch20 : Nx2 tf.float32 tensor
                The tensor containing the localizations.
            d_grid : float, optional
                The precission of the grid we want to plot in between the
                ControlPoints. The default is .1.
            lines_per_CP : int, optional
                The number of lines we want to plot in between the grids. 
                Works best if even. The default is 1.
            locs_markersize : float, optional
                The size of the markers of the localizations. The default is 10.
            CP_markersize : float, optional
                The size of the markers of the Controlpoints. The default is 8.
                
            Returns
            -------
            None.
        
            '''
            print('Plotting the Mapping Grid...')
            if ch1 is None:
                self.center_image()
                ch1=self.ch1.pos
                ch2=self.ch2.pos
                ch20=self.ch20.pos
            if edge_grids is None: edge_grids=self.edge_grid
            
            x1_min = np.min([np.min(tf.reduce_min(tf.floor(ch1[:,0]))),
                                  np.min(tf.reduce_min(tf.floor(ch2[:,0])))])/gridsize
            x2_min = np.min([np.min(tf.reduce_min(tf.floor(ch1[:,1]))),
                                  np.min(tf.reduce_min(tf.floor(ch2[:,1])))])/gridsize
            x1_max = np.max([np.max(tf.reduce_max(tf.floor(ch1[:,0]))),
                                  np.max(tf.reduce_max(tf.floor(ch2[:,0])))])/gridsize
            x2_max = np.max([np.max(tf.reduce_max(tf.floor(ch1[:,1]))),
                                  np.max(tf.reduce_max(tf.floor(ch2[:,1])))])/gridsize
        
            x1_grid = tf.range(x1_min-edge_grids, x1_max+edge_grids+1, dtype=tf.float32)
            x2_grid = tf.range(x2_min-edge_grids, x2_max+edge_grids+1, dtype=tf.float32)
            if model is not None: ControlPoints = model(tf.stack(tf.meshgrid(x1_grid, x2_grid), axis=-1))
            else: ControlPoints = tf.stack(tf.meshgrid(x1_grid, x2_grid), axis=-1)
            
            # plotting the localizations
            plt.figure()
            plt.scatter(ch20[:,0],ch20[:,1], c='green', marker='.', s=locs_markersize, label='Original')
            plt.scatter(ch1[:,0],ch1[:,1], c='red', marker='.', s=locs_markersize, label='Target')
            if plotarrows:
                for i in range(ch1.shape[0]):
                    plt.arrow(ch20[i,0],ch20[i,1], ch2[i,0]-ch20[i,0], ch2[i,1]-ch20[i,1], width=.02, 
                              length_includes_head=True, facecolor='red', edgecolor='red')
            if plotmap: 
                plt.scatter(ch2[:,0],ch2[:,1], c='blue', marker='.', s=locs_markersize, label='Mapped')
            
            # plotting the ControlPoints
            plt.scatter(ControlPoints[:,:,0]*gridsize, ControlPoints[:,:,1]*gridsize,
                        c='b', marker='d', s=CP_markersize)
            
            ## Horizontal Grid
            x1_grid = tf.range(x1_min-edge_grids, tf.math.ceil(x1_max)+edge_grids+d_grid, delta=d_grid, dtype=tf.float32)
            x2_grid = tf.range(x2_min-edge_grids, tf.math.ceil(x2_max)+edge_grids+d_grid, delta=1/Ngrids, dtype=tf.float32)
            Grid = tf.reshape(tf.stack(tf.meshgrid(x1_grid, x2_grid), axis=-1) , (-1,2)) 
            if model is not None: Grid = model( Grid ) * gridsize
            else: Grid = Grid*gridsize
            (nn, i,j)=(x1_grid.shape[0],0,0)
            while i<Grid.shape[0]:
                if j%Ngrids==0:
                    plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='b')
                else:
                    plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='c')
                i+=nn
                j+=1

            ## Vertical Grid
            x1_grid = tf.range(x1_min-edge_grids, tf.math.ceil(x1_max)+edge_grids+d_grid, delta=1/Ngrids, dtype=tf.float32)
            x2_grid = tf.range(x2_min-edge_grids, tf.math.ceil(x2_max)+edge_grids+d_grid, delta=d_grid, dtype=tf.float32)
            Grid = tf.gather(tf.reshape(tf.stack(tf.meshgrid(x2_grid, x1_grid), axis=-1) , (-1,2)), [1,0], axis=1)
            if model is not None: Grid = model( Grid ) * gridsize
            else: Grid = Grid*gridsize
            (nn, i,j)=(x2_grid.shape[0],0,0)
            while i<Grid.shape[0]:
                if j%Ngrids==0:
                    plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='b')
                else:
                    plt.plot(Grid[i:i+nn,0], Grid[i:i+nn,1], c='c')
                i+=nn
                j+=1
            
            plt.legend()
            plt.tight_layout()
