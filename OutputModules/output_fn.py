# output_text.py
"""
Created on Thu Apr 22 14:26:22 2021

@author: Mels
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import copy
import tensorflow as tf

import OutputModules.generate_image as generate_image
from Align_Modules.generate_neighbours import KNN

#%% error handling of batches
def Info_batch(N1, N2, coupled):
    '''
    Handles the error if batches are sufficient large

    Parameters
    ----------
    N : int
        total amount of points.
    num_batches : 2 int array
        containing the amount of [x1,x2] batches.
    batch_size : int
        max number of points per batch.
    Batch_on : bool, optional
        Are batches used. The default is True.

    Returns
    -------
    None.

    '''
    N = np.max([N1, N2])
    print('\nI: the total system contains', N, ' points. The setup seems to be OK',
          '\nNote that for big N, batches shoud be used.\n')
    if N1!=N2 and coupled:
        print('\nError : System is coupled but Channel B contains',N2,'points and Channel A',N1)
    time.sleep(2)


#%% Error distribution 
def errorHist(ch1, ch2, ch2_map, ch1_copy2=None, nbins=30, plot_on=True, direct=False):
    '''
    Generates a histogram showing the distribution of distances between coupled points

    Parameters
    ----------
    ch1 : Nx2
        The localizations of channel 1.
    ch2 , ch2m : Nx2
        The localizations of channel 2 and the mapped channel 2. The indexes 
        of should be one-to-one with channel 1
    ch1_copy2 : Nx2, optional
        The localizations that are coupled with ch2. The default is None
    nbins : list int, optional
        The number of bins. The default is 30.
    plot_on : bool, optional
        Do we want to plot. The default is True.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    if ch1_copy2 is None: ch1_copy2=ch1
    
    # calculate the bars and averages via coupled method
    if direct:
        dist1 = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
        dist2 = np.sqrt( np.sum( ( ch1_copy2 - ch2_map )**2, axis = 1) )
        
        avg1 = np.average(dist1)
        avg2 = np.average(dist2)

    # calculate the bars and averages via KNN method
    else:
        idx1 = KNN(ch1, ch2, 1)
        idx2 = KNN(ch1_copy2, ch2_map, 1)
        
        dist1 = np.sqrt( np.sum( ( ch1 - ch2[idx1,:][:,0,:] )**2, axis = 1) )
        dist2 = np.sqrt( np.sum( ( ch1_copy2 - ch2_map[idx2,:][:,0,:] )**2, axis = 1) )
        
        avg1 = np.average(dist1)
        avg2 = np.average(dist2)
            
    # Plotting
    if plot_on:        
        # Plotting the histogram
        fig, (ax1, ax2) = plt.subplots(2)
        n1 = ax1.hist(dist1+.25, label='Original', alpha=.8, edgecolor='red', color='tab:blue', bins=nbins)
        n2 = ax1.hist(dist2, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
        n2 = ax2.hist(dist2, label='Mapped', alpha=.8, edgecolor='red', color='tab:orange', bins=nbins)
            
        # Plotting the averages as vlines
        ymax = np.max([np.max(n1[0]), np.max(n2[0])]) + 50
        ax1.vlines(avg1, color='purple', ymin=0, ymax=ymax, label=('avg original = '+str(round(avg1,2))))
        ax1.vlines(avg2, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg2,2))))
        ax2.vlines(avg2, color='green', ymin=0, ymax=ymax, label=('avg mapped = '+str(round(avg2,2))))
            
        # Some extra plotting parameters
        ax1.set_title('Comparisson')
        ax1.set_ylim([0,ymax])
        ax1.set_xlim(0)
        ax1.set_xlabel('distance [nm]')
        ax1.set_ylabel('# of localizations')
        ax1.legend()
                
        ax2.set_title('Zoomed in on Mapping Error')
        ax2.set_ylim([0,ymax])
        ax2.set_xlim(0)
        ax2.set_xlabel('distance [nm]')
        ax2.set_ylabel('# of localizations')
        ax2.legend()
        
        fig.show()
    else:
        fig=None
        ax1=None
        ax2=None
    
    return avg1, avg2, fig, (ax1, ax2)


#%% Error distribution over FOV 
def errorFOV(ch1, ch2, ch2_map, ch1_copy2=None, plot_on=True, direct=False):
    '''
    Generates a FOV distribution of distances between coupled points

    Parameters
    ----------
    ch1 : Nx2
        The localizations of channel 1.
    ch2 , ch2m : Nx2
        The localizations of channel 2 and the mapped channel 2. The indexes 
        of should be one-to-one with channel 1
    ch1_copy2 : Nx2, optional
        The localizations that are coupled with ch2. The default is None
    plot_on : bool, optional
        Do we want to plot. The default is True.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    if ch1_copy2 is None: ch1_copy2=ch1
    
    if direct:
        r1 = np.sqrt(np.sum(ch1**2,1))
        r2 = np.sqrt(np.sum(ch1_copy2**2,1))
        error1 = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
        error2 = np.sqrt( np.sum( ( ch1_copy2 - ch2_map )**2, axis = 1) )
        
        avg1 = np.average(error1)
        avg2 = np.average(error2)
            
    else:
        r1 = np.sqrt(np.sum(ch1**2,1))
        r2 = np.sqrt(np.sum(ch1_copy2**2,1))
        idx1 = KNN(ch1, ch2, 1)
        idx2 = KNN(ch1_copy2, ch2_map, 1)
        error1 = np.sqrt( np.sum( ( ch1 - ch2[idx1,:][:,0,:] )**2, axis = 1) )
        error2 = np.sqrt( np.sum( ( ch1_copy2 - ch2_map[idx2,:][:,0,:] )**2, axis = 1) )
        
        avg1 = np.average(error1)
        avg2 = np.average(error2)
        
    if plot_on:        
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(r1, error1, 'b.', alpha=.4, label='Original error')
        ax1.plot(r2, error2, 'r.', alpha=.4, label='Mapped error')
        ax2.plot(r2, error2, 'r.', alpha=.4, label='Mapped error')
        
        # Plotting the averages as hlines
        xmax= np.max((np.max(r1),np.max(r2)))+50
        ax1.hlines(avg1, color='purple', xmin=0, xmax=xmax, label=('average original = '+str(round(avg1,2))))
        ax1.hlines(avg2, color='green', xmin=0, xmax=xmax, label=('average mapped = '+str(round(avg2,2))))
        ax2.hlines(avg2, color='green', xmin=0, xmax=xmax, label=('average mapped = '+str(round(avg2,2))))
            
        # Some extra plotting parameters
        ax1.set_title('Comparisson')
        ax1.set_ylim(0)
        ax1.set_xlim([0,xmax])
        ax1.set_xlabel('FOV [nm]')
        ax1.set_ylabel('Absolute Error')
        ax1.legend()
                
        ax2.set_title('Zoomed in on Mapping Error')
        ax2.set_ylim(0)
        ax2.set_xlim([0,xmax])
        ax2.set_xlabel('FOV [nm]')
        ax2.set_ylabel('Absolute Error')
        ax2.legend()
        
        fig.show()

    
    return avg1, avg2, fig, (ax1, ax2)


#%% Error distribution over FOV complete
def errorFOV_complete(ch1, ch2, direct=False, precision = 100):
    '''
    Generates a FOV distribution of distances between coupled points

    Parameters
    ----------
    ch1 , ch2 : Nx2
        The localizations of channel 1 and 2.
        of should be one-to-one with channel 1
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    if direct:
        error = tf.reduce_sum(ch1 - ch2, axis=1)
    else:
        idx1 = KNN(ch1, ch2, 1)
        error = tf.reduce_sum(ch1 - ch2[idx1,:][:,0,:], axis=1)
    
    ch1 = ch1/precision
    ch2 = ch2/precision
        
    bounds = np.empty([2,2], dtype = float) 
    bounds[0,0] = np.min(ch1[:,0])
    bounds[0,1] = np.max(ch1[:,0])
    bounds[1,0] = np.min(ch1[:,1])
    bounds[1,1] = np.max(ch1[:,1])
    
    error_mat = generate_image.generate_matrix(ch1, bounds, error=error)
    
    plt.figure()
    plt.imshow(error_mat)
    plt.colorbar()
    
    return error_mat
        


#%% Plotting the grid
def plot_grid(ch1, ch2, ch2_map, mods, gridsize=50, d_grid=.1, lines_per_CP=1, 
              locs_markersize=10, CP_markersize=8, grid_markersize=3, grid_opacity=1,
              sys_param=None): 
    '''
    Plots the grid and the shape of the grid in between the Control Points

    Parameters
    ----------
    ch1 , ch2 , ch2_map : Nx2 tf.float32 tensor
        The tensor containing the localizations.
    mods : Models() Class
        The Model which has been trained on the dataset.
    gridsize : float, optional
        The size of the grid used in mods. The default is 50.
    d_grid : float, optional
        The precission of the grid we want to plot in between the
        Control Points. The default is .1.
    locs_markersize : float, optional
        The size of the markers of the localizations. The default is 10.
    CP_markersize : float, optional
        The size of the markers of the Controlpoints. The default is 8.
    grid_markersize : float, optional
        The size of the markers of the grid. The default is 3.
    grid_opacity : float, optional
        The opacity of the grid. The default is 1.
    lines_per_CP : int, optional
        The number of lines we want to plot in between the grids. 
        Works best if even. The default is 1.
    sys_params : list, optional
        List containing the size of the system. The optional is None,
        which means it will be calculated by hand

    Returns
    -------
    None.

    '''
    print('Plotting the Spline Grid...')
    if sys_param is None:
        x1_min = tf.reduce_min(tf.floor(ch2[:,0]/gridsize))
        x2_min = tf.reduce_min(tf.floor(ch2[:,1]/gridsize))
        x1_max = tf.reduce_max(tf.floor(ch2[:,0]/gridsize))
        x2_max = tf.reduce_max(tf.floor(ch2[:,1]/gridsize))
    else:
        x1_min = tf.floor(sys_param[0,0]/gridsize)
        x2_min = tf.floor(sys_param[0,1]/gridsize)
        x1_max = tf.floor(sys_param[1,0]/gridsize)
        x2_max = tf.floor(sys_param[1,1]/gridsize)
    
    ## Creating the horizontal grid
    grid_tf = []
    marker = []
    
    x1_grid = tf.range(x1_min, x1_max+1, d_grid)
    x2_grid = (x2_min) * tf.ones(x1_grid.shape[0], dtype=tf.float32)
    while x2_grid[0] < x2_max+1.8:
        # Mark the grid lines from the inbetween grid lines
        if x2_grid[0]%1<.05 or x2_grid[0]%1>.95:            marker.append(np.ones(x1_grid.shape))
        else:        marker.append(np.zeros(x1_grid.shape))
        
        # Create grid
        grid_tf.append(tf.concat((x1_grid[:,None], x2_grid[:,None]), axis=1))
        x2_grid +=  np.round(1/lines_per_CP,2)
        
    # Creating the right grid line
    grid_tf.append(tf.concat(( x1_grid[:,None], 
                              (x2_max+.99)*tf.ones([x1_grid.shape[0],1], dtype=tf.float32),
                              ), axis=1))
    marker.append(np.ones(x1_grid.shape))
    
    # Creating the vertical grid
    x2_grid = tf.range(x2_min, x2_max+1, d_grid)
    x1_grid = (x1_min) * tf.ones(x2_grid.shape[0], dtype=tf.float32)
    while x1_grid[0] < x1_max+1.8:
        # Mark the grid lines from the inbetween grid lines
        if x1_grid[0]%1<.05 or x1_grid[0]%1>.95:            marker.append(np.ones(x1_grid.shape))
        else:        marker.append(np.zeros(x1_grid.shape))
        
        # Create grid
        grid_tf.append(tf.concat((x1_grid[:,None], x2_grid[:,None]), axis=1))
        x1_grid += np.round(1/lines_per_CP,2)
        
    # Creating the upper grid line
    grid_tf.append(tf.concat(( (x1_max+.99)*tf.ones([x1_grid.shape[0],1], dtype=tf.float32),
                                  x2_grid[:,None]), axis=1))
    marker.append(np.ones(x1_grid.shape))
        
    # Adding to get the original grid 
    grid_tf = tf.concat(grid_tf, axis=0)
    marker = tf.concat(marker, axis=0)
    CP_idx = tf.cast(tf.stack(
            [( grid_tf[:,0]-tf.reduce_min(tf.floor(grid_tf[:,0]))+2)//1 , 
             ( grid_tf[:,1]-tf.reduce_min(tf.floor(grid_tf[:,1]))+2)//1 ], 
            axis=1), dtype=tf.int32)
    
    # transforming the grid
    mods_temp = copy(mods.model)
    mods_temp.reset_CP(CP_idx)
    grid_tf = mods_temp.transform_vec(grid_tf)
    
    # plotting the localizations
    plt.figure()
    plt.plot(ch2_map[:,0],ch2_map[:,1], color='red', marker='.', linestyle='',
             markersize=locs_markersize, label='Mapped CH2')
    plt.plot(ch2[:,0],ch2[:,1], color='orange', marker='.', linestyle='', 
             alpha=.7, markersize=locs_markersize-2, label='Original CH2')
    plt.plot(ch1[:,0],ch1[:,1], color='green', marker='.', linestyle='', 
             markersize=locs_markersize, label='Original CH1')
    
    # spliting grid from inbetween grid
    if lines_per_CP != 1:
        marker_idx1=np.argwhere(marker==1)
        marker_idx2=np.argwhere(marker==0)
        grid_tf1 = tf.gather_nd(grid_tf, marker_idx1)
        grid_tf2 = tf.gather_nd(grid_tf, marker_idx2)
        #print(grid_tf.shape, grid_tf1.shape, grid_tf2.shape)
        
        # plotting the gridlines
        plt.plot(grid_tf1[:,0]*gridsize,grid_tf1[:,1]*gridsize, 'b.',
                 markersize=grid_markersize, alpha=grid_opacity)
        plt.plot( mods.model.CP_locs[:,:,0]*gridsize,  mods.model.CP_locs[:,:,1]*gridsize, 
                 'b+', markersize=CP_markersize)

        # plotting the inbetween gridlines
        plt.plot(grid_tf2[:,0]*gridsize,grid_tf2[:,1]*gridsize, 'c.',
                 markersize=grid_markersize, alpha=grid_opacity)
    
    else:
        # plotting the gridlines
        plt.plot(grid_tf[:,0]*gridsize,grid_tf[:,1]*gridsize, 'b.',
                 markersize=grid_markersize, alpha=grid_opacity)
        plt.plot( mods.model.CP_locs[:,:,0]*gridsize,  mods.model.CP_locs[:,:,1]*gridsize, 
                 'b+', markersize=CP_markersize)
    plt.legend()