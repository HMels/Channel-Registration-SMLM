# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:03:46 2021

@author: Mels
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class Analysis:
    def __init__(self, AlignModel=None,ch1=None, ch2=None, ch2_original=None, coupled=False):
        if AlignModel is None and ch1 is None and ch2 is None: 
            raise Exception('No Data selected to Analyse, please input the Dataset or the 2 channels (ch1,ch2) seperately!')
        
        ## build variables
        if AlignModel is not None:
            self.ch1 = AlignModel.ch1.pos
            self.ch2 = AlignModel.ch2.pos
            self.ch2_original = AlignModel.ch2_original.pos
            self.coupled = AlignModel.coupled
        else:
            self.ch1=ch1
            self.ch2=ch2
            self.ch2_original=ch2_original
            self.coupled=coupled
            
    
    #%% error_fn
    def couple_dataset(self, ch1=None, ch2=None, maxDist=50, Filter=True):
    # couples dataset with a simple iterative nearest neighbour method
        print('Coupling datasets with an iterative method...')
        if ch1 is None: 
            ch1=self.ch1
            ch2=self.ch2
            
        locsA=[]
        locsB=[]
        for i in range(ch1.shape[0]):
            dists = np.sqrt(np.sum((ch1[i,:]-ch2)**2,1))
            if not Filter or np.min(dists)<maxDist:
                locsA.append( ch1[i,:] )
                locsB.append( ch2[np.argmin(dists),:] )
        return np.array(locsA), np.array(locsB)
        
        
    def ErrorDist(self, ch1, ch2):
    # Generates the error, average and radius
        dist = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
        return dist, np.average(dist), np.sqrt(np.sum(ch1**2,1))
    
    
    
    #%% Plotting the error
    def ErrorPlot(self, nbins=30):
        ## Coupling Datasets if not done already
        if not self.coupled: 
            ch1, ch2 = self.couple_dataset(self.ch1, self.ch2)
            if self.ch2_original is not None: 
                ch1_original, ch2_original = self.couple_dataset(self.ch1, self.ch2_original)
        else:
            ch1=self.ch1
            ch2=self.ch2
            ch1_original=self.ch1
            ch2_original=self.ch2_original
            
        
        # Calculating the error
        dist1, avg1, r1 = self.ErrorDist(ch1, ch2)
        if self.ch2_original is not None: 
            dist2, avg2, r2 = self.ErrorDist(ch1_original, ch2_original)
        
        
        ## Plotting
        if self.ch2_original is not None: fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(2,2)
        else: fig, (ax1, ax2) = plt.subplots(2)
          
        # plotting the histogram
        n1 = ax1.hist(dist1, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        if self.ch2_original is not None:
            n1 = ax3.hist(dist1+.25, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
            n2 = ax3.hist(dist2, label='Original', alpha=.8, edgecolor='red', color='tab:blue', bins=nbins)
        else:
            n2=[0]
        ymax = np.max([np.max(n1[0]), np.max(n2[0])]) + 50
            
        # plotting the FOV
        ax2.plot(r1, dist1, 'r.', alpha=.4, label='Mapped error')
        if self.ch2_original is not None:
            ax4.plot(r1, dist1, 'r.', alpha=.4, label='Mapped error')
            ax4.plot(r2, dist2, 'b.', alpha=.4, label='Original error') 
        else:
            r2=0
        xmax= np.max((np.max(r1),np.max(r2)))+50
        
        # Plotting the averages as vlines
        ax1.vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg1,2))))
        if self.ch2_original is not None:
            ax3.vlines(avg2, color='purple', ymin=0, ymax=ymax, label=('avg original = '+str(round(avg2,2))))
            ax3.vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg1,2))))
          
        # Plotting the averages as hlines
        ax2.hlines(avg1, color='green', xmin=0, xmax=xmax, label=('average mapped = '+str(round(avg1,2))))
        if self.ch2_original is not None:
            ax4.hlines(avg2, color='purple', xmin=0, xmax=xmax, label=('average original = '+str(round(avg2,2))))
            ax4.hlines(avg1, color='green', xmin=0, xmax=xmax, label=('average mapped = '+str(round(avg1,2))))
        
        
        # Some extra plotting parameters
        ax1.set_title('Zoomed in on Mapping Error')
        ax1.set_ylim([0,ymax])
        ax1.set_xlim(0)
        ax1.set_xlabel('distance [nm]')
        ax1.set_ylabel('# of localizations')
        ax1.legend()
        
        ax2.set_title('Zoomed in on Mapping Error')
        ax2.set_ylim(0)
        ax2.set_xlim([0,xmax])
        ax2.set_xlabel('FOV [nm]')
        ax2.set_ylabel('Absolute Error')
        ax2.legend()
        
        if self.ch2_original is not None:
            ax3.set_title('Comparisson')
            ax3.set_ylim([0,ymax])
            ax3.set_xlim(0)
            ax3.set_xlabel('distance [nm]')
            ax3.set_ylabel('# of localizations')
            ax3.legend()
        
            ax4.set_title('Comparisson')
            ax4.set_ylim(0)
            ax4.set_xlim([0,xmax])
            ax4.set_xlabel('FOV [nm]')
            ax4.set_ylabel('Absolute Error')
            ax4.legend()
           
        fig.show()
        if self.ch2_original is not None: 
            print('The original model had an average error of',avg2,'nm\nThe mapped model has an average error of',avg1,'nm')
            return avg1, avg2, fig, (ax3, ax1, ax4, ax2)
        else: 
            print('The mapped model has an average error of',avg1,'nm')
            return avg1, fig, (ax1, ax2)


    #%% plotting the error in a [x1, x2] plot like in the paper        
    def ErrorPlotImage(self, other=None, maxDist=30, ps=5, cmap='seismic'):
        ## Coupling Dataset1 if not done already
        if not self.coupled: 
            ch11, ch12 = self.couple_dataset(self.ch1, self.ch2, maxDist=maxDist, Filter=True)
            ch11=self.ch1
            ch12=self.ch2
        else:
            ch11=self.ch1
            ch12=self.ch2
        dist = ch11-ch12
        if dist.shape==(0,): raise ValueError('No neighbours found for channel 1')
            
        ## Coupling Dataset2 if not done already
        if other is not None:
            if not other.coupled: 
                ch21, ch22 = other.couple_dataset(other.ch1, other.ch2, maxDist=maxDist, Filter=True)
            else:
                ch21=other.ch1
                ch22=other.ch2
            dist1 = ch21-ch22
            if dist1.shape==(0,): raise ValueError('No neighbours found for channel 2')
            
            fig, ax = plt.subplots(2,2)
            ax[0][0].scatter(ch11[:,0], ch11[:,1], s=ps, c=dist[:,0], cmap=cmap)
            ax[0][0].set_xlabel('x-position [nm]')
            ax[0][0].set_ylabel('Set 1 Fiducials\ny-position [nm]')
            norm=mpl.colors.Normalize(vmin=np.min(dist[:,0]), vmax=np.max(dist[:,0]), clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-offset [nmn]', ax=ax[0][0])
            
            ax[0][1].scatter(ch11[:,0], ch11[:,1], s=ps, c=dist[:,1], cmap=cmap)
            ax[0][1].set_xlabel('x-position [nm]')
            ax[0][1].set_ylabel('y-position [nmn]')
            norm=mpl.colors.Normalize(vmin=np.min(dist[:,1]), vmax=np.max(dist[:,1]), clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-offset [nm]', ax=ax[0][1])
        
            ax[1][0].scatter(ch21[:,0], ch21[:,1], s=ps, c=dist1[:,0], cmap=cmap)
            ax[1][0].set_xlabel('x-position [nm]')
            ax[1][0].set_ylabel('Set 2 Fiducials\ny-position [nm]')
            norm=mpl.colors.Normalize(vmin=np.min(dist1[:,0]), vmax=np.max(dist1[:,0]), clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-offset [nm]', ax=ax[1][0])
            
            ax[1][1].scatter(ch21[:,0], ch21[:,1], s=ps, c=dist1[:,1], cmap=cmap)
            ax[1][1].set_xlabel('x-position [nm]')
            ax[1][1].set_ylabel('y-position [nm]')
            norm=mpl.colors.Normalize(vmin=np.min(dist1[:,1]), vmax=np.max(dist1[:,1]), clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-offset [nm]', ax=ax[1][1])
            
        else:
            fig, ax = plt.subplots(1,2)
            ax[0].scatter(ch11[:,0], ch11[:,1], s=ps, c=dist[:,0], cmap=cmap)
            ax[0].set_xlabel('x-position [nm]')
            ax[0].set_ylabel('Set 1 Fiducials\ny-position [nm]')
            norm=mpl.colors.Normalize(vmin=np.min(dist[:,0]), vmax=np.max(dist[:,0]), clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-offset [nmn]', ax=ax[0])
            
            ax[1].scatter(ch11[:,0], ch11[:,1], s=ps, c=dist[:,1], cmap=cmap)
            ax[1].set_xlabel('x-position [nm]')
            ax[1].set_ylabel('y-position [nmn]')
            norm=mpl.colors.Normalize(vmin=np.min(dist[:,1]), vmax=np.max(dist[:,1]), clip=False)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-offset [nm]', ax=ax[1])
        
        
            
    
        
    #%% Channel to matrix fn
    def isin_domain(self, pos):
    # checks if pos is within bounds
        return ( pos[0] > self.bounds[0,0] and pos[1] > self.bounds[1,0] and 
                pos[0] < self.bounds[0,1] and pos[1] < self.bounds[1,1] )
    
    
    def generate_channel(self, precision=10):
    # Generates the channels as matrix
        print('Generating Channels as matrix')       
    
        # normalizing system
        locs1 = self.ch1  / precision
        locs2 = self.ch2  / precision
        if self.ch2_original is not None: locs2_original = self.ch2_original  / precision
        else: locs2_original=locs2
        
        # calculate bounds of the system
        self.precision=precision
        self.bounds = np.empty([2,2], dtype = float) 
        self.bounds[0,0] = np.min([ np.min(locs1[:,0]), np.min(locs2[:,0]), np.min(locs2_original[:,0]) ])
        self.bounds[0,1] = np.max([ np.max(locs1[:,0]), np.max(locs2[:,0]), np.max(locs2_original[:,0]) ])
        self.bounds[1,0] = np.min([ np.min(locs1[:,1]), np.min(locs2[:,1]), np.min(locs2_original[:,1]) ])
        self.bounds[1,1] = np.max([ np.max(locs1[:,1]), np.max(locs2[:,1]), np.max(locs2_original[:,1]) ])
        self.size_img = np.round( (self.bounds[:,1] - self.bounds[:,0]) , 0).astype('int')           
        self.axis = np.array([ self.bounds[1,:], self.bounds[0,:]]) * self.precision
        self.axis = np.reshape(self.axis, [1,4])[0]
        
        # generating the matrices to be plotted
        self.channel1 = self.generate_matrix(locs1)
        self.channel2 = self.generate_matrix(locs2)
        if self.ch2_original is not None: self.channel2_original = self.generate_matrix(locs2_original)
        
        
    def generate_matrix(self, locs):
    # takes the localizations and puts them in a channel
        channel = np.zeros([self.size_img[0], self.size_img[1]], dtype = int)
        for i in range(locs.shape[0]):
            loc = np.round(locs[i,:],0).astype('int')
            if self.isin_domain(loc):
                loc -= np.round(self.bounds[:,0],0).astype('int') # place the zero point on the left
                channel[loc[0]-1, loc[1]-1] = 1
        return channel
    
    
    #%% Plotting Channels
    def plot_channel(self):
        print('Plotting...')
        
        # plotting all channels
        plt.figure()
        if self.ch2_original is not None: plt.subplot(131)
        else: plt.subplot(121)
        plt.imshow(self.channel1, extent = self.axis)
        plt.xlabel('x2')
        plt.ylabel('x1')
        plt.title('original channel 1')
        
        if self.ch2_original is not None: plt.subplot(132)
        else: plt.subplot(122)
        plt.imshow(self.channel2, extent = self.axis)
        plt.xlabel('x2')
        plt.ylabel('x1')
        plt.title('mapped channel 2')
        
        if self.ch2_original is not None: 
            plt.subplot(133)
            plt.imshow(self.channel2_original, extent = self.axis)
            plt.xlabel('x2')
            plt.ylabel('x1')
            plt.title('original channel 2')
        
        
    def plot_1channel(self, channel1=None):
        if channel1 is None: channel1=self.channel1
        # plotting all channels
        plt.figure()
        plt.imshow(np.rot90(channel1))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Single Channel view')