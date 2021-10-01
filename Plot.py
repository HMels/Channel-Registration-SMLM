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
import matplotlib as mpl

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
        if not self.coupled: raise Exception('Dataset should first be coupled before registration errors can be derived!')
        pos1_original=self.ch1.pos
        pos2_original=self.ch2_original.pos
        pos1=self.ch1.pos
        pos2=self.ch2.pos
        
        # Calculating the error
        dist1, avg1, r1 = self.ErrorDist(pos1, pos2)
        if self.ch2_original.pos is not None: 
            dist2, avg2, r2 = self.ErrorDist(pos1_original, pos2_original)
        
        
        ## Plotting
        if self.ch2_original.pos is not None: fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(2,2)
        else: fig, (ax1, ax2) = plt.subplots(2)
          
        # plotting the histogram
        n1 = ax1.hist(dist1, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        if self.ch2_original.pos is not None:
            n1 = ax3.hist(dist1, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
            n2 = ax3.hist(dist2, label='Original', alpha=.8, edgecolor='red', color='tab:blue', bins=nbins)
        else:
            n2=[0]
        ymax = np.max([np.max(n1[0]), np.max(n2[0])])*1.1
            
        # plotting the FOV
        ax2.plot(r1, dist1, 'r.', alpha=.4, label='Mapped error')
        if self.ch2_original.pos is not None:
            ax4.plot(r1, dist1, 'r.', alpha=.4, label='Mapped error')
            ax4.plot(r2, dist2, 'b.', alpha=.4, label='Original error') 
        else:
            r2=np.array(0)
        xmax= np.max((np.max(r1),np.max(r2)))*1.1
        
        # Plotting the averages as vlines
        ax1.vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg1,2))))
        if self.ch2_original.pos is not None:
            ax3.vlines(avg2, color='purple', ymin=0, ymax=ymax, label=('avg original = '+str(round(avg2,2))))
            ax3.vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg1,2))))
          
        # Plotting the averages as hlines
        ax2.hlines(avg1, color='green', xmin=0, xmax=xmax, label=('average mapped = '+str(round(avg1,2))))
        if self.ch2_original.pos is not None:
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
        
        if self.ch2_original.pos is not None:
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
        if self.ch2_original.pos is not None: 
            print('The original model had an average absolute error of',avg2,'nm\nThe mapped model has an average absolute error of',avg1,'nm')
            return avg1, avg2, fig, (ax3, ax1, ax4, ax2)
        else: 
            print('The mapped model has an average absolute error of',avg1,'nm')
            return avg1, fig, (ax1, ax2)


    def ErrorDistribution(self, nbins=30, FrameLinking=False):
    # just plots the error distribution after mapping
        if not self.coupled: raise Exception('Dataset should first be coupled before registration errors can be derived!')
        pos1=self.ch1.pos
        pos2=self.ch2.pos
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
        
        
    def ErrorDistribution_xy(self, nbins=30, xlim=31, FrameLinking=False):
        if not self.coupled: raise Exception('Dataset should first be coupled before registration errors can be derived!')
        pos1=self.ch1.pos
        pos2=self.ch2.pos
            
        
        distx=pos1[:,0]-pos2[:,0]
        avgx = np.average(distx)
        stdx = np.std(distx)
        disty=pos1[:,1]-pos2[:,1]
        avgy = np.average(disty)
        stdy = np.std(disty)
            
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        
        nx = ax[0].hist(distx, label=(r'$\mu$ = '+str(round(avgx,2))+'nm, $\sigma$ = '+str(round(stdx,2))+'nm'),
                        alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        ny = ax[1].hist(disty, label=(r'$\mu$ = '+str(round(avgy,2))+'nm, $\sigma$ = '+str(round(stdy,2))+'nm'),
                        alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        
        ymax = np.max([np.max(nx[0]),np.max(ny[0])])*1.1
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
        

    #%% plotting the error in a [x1, x2] plot like in the paper        
    def ErrorPlotImage(self, other=None, maxDist=30, ps=5, cmap='seismic', FrameLinking=False):
        ## Coupling Dataset1 if not done already
        if not self.coupled: raise Exception('Dataset should first be coupled before registration errors can be derived!')
        dist = self.ch1.pos-self.ch2.pos
        if dist.shape==(0,): raise ValueError('No neighbours found for channel 1')
            
        ## Coupling Dataset2 if not done already
        if other is not None:
            if not self.coupled: raise Exception('Dataset should first be coupled before registration errors can be derived!')
            dist1 = other.ch1.pos-other.ch2.pos
            if dist1.shape==(0,): raise ValueError('No neighbours found for channel 2')
            
            vmin=np.min((np.min(dist[:,0]),np.min(dist1[:,0]),np.min(dist[:,1]),np.min(dist1[:,1])))
            vmax=np.max((np.max(dist[:,0]),np.max(dist1[:,0]),np.max(dist[:,1]),np.max(dist1[:,1])))
            
            
            fig, ax = plt.subplots(2,2)
            ax[0][0].scatter(self.ch1.pos[:,0]/1000, self.ch1.pos[:,1]/1000, s=ps, c=dist[:,0], cmap=cmap, vmin=vmin, vmax=vmax)
            #ax[0][0].set_xlabel('x-position [\u03bcm]')
            ax[0][0].set_ylabel('Set 1 Fiducials\ny-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-error [nm]', ax=ax[0][0])
            ax[0][0].set_aspect('equal', 'box')
            
            ax[0][1].scatter(self.ch1.pos[:,0]/1000, self.ch1.pos[:,1]/1000, s=ps, c=dist[:,1], cmap=cmap, vmin=vmin, vmax=vmax)
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
            ax[0].scatter(self.ch1.pos[:,0]/1000, self.ch1.pos[:,1]/1000, s=ps, c=dist[:,0], cmap=cmap)
            ax[0].set_xlabel('x-position [\u03bcm]')
            ax[0].set_ylabel('Batch 1\ny-position [\u03bcm]')
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-error [nm]', ax=ax[0])
            ax[0].set_aspect('equal', 'box')
            
            ax[1].scatter(self.ch1.pos[:,0]/1000, self.ch1.pos[:,1]/1000, s=ps, c=dist[:,1], cmap=cmap)
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
    
    
    def generate_channel(self, precision=10):
    # Generates the channels as matrix
        print('Generating Channels as matrix')       
    
        # normalizing system
        locs1 = self.ch1.pos  / precision
        locs2 = self.ch2.pos  / precision
        if self.ch2_original.pos is not None: locs2_original = self.ch2_original.pos  / precision
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
        self.channel1 = self.generate_matrix(locs1)
        self.channel2 = self.generate_matrix(locs2)
        if self.ch2_original.pos is not None: self.channel2_original = self.generate_matrix(locs2_original)
        
        
    def generate_matrix(self, locs):
    # takes the localizations and puts them in a channel
        channel = np.zeros([self.size_img[0]+1, self.size_img[1]+1], dtype = int)
        for i in range(locs.shape[0]):
            loc = np.round(locs[i,:],0).astype('int')
            if self.isin_domain(loc):
                loc -= np.round(self.bounds[:,0],0).astype('int') # place the zero point on the left
                channel[loc[0], loc[1]] = 1
        return channel
    
    
    #%% Plotting Channels
    def plot_channel(self):
        print('Plotting...')
        
        # plotting all channels
        plt.figure()
        if self.ch2_original.pos is not None: plt.subplot(131)
        else: plt.subplot(121)
        plt.imshow(self.channel1, extent = self.axis)
        plt.xlabel('x2')
        plt.ylabel('x1')
        plt.title('original channel 1')
        plt.tight_layout()
        
        if self.ch2_original.pos is not None: plt.subplot(132)
        else: plt.subplot(122)
        plt.imshow(self.channel2, extent = self.axis)
        plt.xlabel('x2')
        plt.ylabel('x1')
        plt.title('mapped channel 2')
        plt.tight_layout()
        
        if self.ch2_original.pos is not None: 
            plt.subplot(133)
            plt.imshow(self.channel2_original, extent = self.axis)
            plt.xlabel('x2')
            plt.ylabel('x1')
            plt.title('original channel 2')
            plt.tight_layout()
        
        
    def plot_1channel(self, channel1=None):
        if channel1 is None: channel1=self.channel1
        # plotting all channels
        plt.figure()
        plt.imshow(np.rot90(channel1))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Single Channel view')
        plt.tight_layout()