# generate_image.py
"""
Created on Mon Apr 19 14:09:07 2021

@author: Mels
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def plot_channel(channel1, channel2, channel3, bounds,  precision):
    print('Plotting...')
    
    axis = np.array([ bounds[1,:], bounds[0,:]]) * precision
    axis = np.reshape(axis, [1,4])[0]
    
    # plotting all channels
    plt.figure()
    plt.subplot(131)
    plt.imshow(channel1, extent = axis)
    plt.title('original channel 1')
    
    plt.subplot(132)
    plt.imshow(channel2, extent = axis)
    plt.title('original channel 2')
    
    plt.subplot(133)
    plt.imshow(channel3, extent = axis)
    plt.title('channel 2 mapped')
    
    
def plot_1channel(channel1, bounds=None, precision=20):
    
    if bounds is None:
        bounds = np.empty([2,2], dtype = float) 
        bounds[0,0] = np.min(channel1[:,0])
        bounds[0,1] = np.max(channel1[:,0])
        bounds[1,0] = np.min(channel1[:,1])
        bounds[1,1] = np.max(channel1[:,1])
        
    axis = np.array([ bounds[1,:], bounds[0,:]]) * precision
    axis = np.reshape(axis, [1,4])[0]
    
    # plotting all channels
    plt.figure()
    plt.imshow(np.rot90(channel1))
    plt.title('original channel 1')


def generate_channel(locs1, locs2, locs3, precision = 10, max_deform = 150):
    '''
    Generates the image of all three figures

    Parameters
    ----------
    locs1, locs2, locs3 : Nx2 float array
        The localizations in nm.
    precision : float, optional
        The amount of precision in nm. The default is 10
    max_deform : float, optional
        The maximum amount of deformation allowed in nm. The default is 150.

    Returns
    -------
    channel1, channel2, channel3 : int array
        Array containing the different channels in matrices, in which 1 means there is 
        one or more localizations 
    axis : 4 float array
        Array containing the axis boundaries for the plots in nm

    '''
    print('Preparing Data for plotting...')
    bounds_check(locs1, locs3, max_deform)
    

    # normalizing system
    locs1 = locs1  / precision
    locs2 = locs2  / precision
    locs3 = locs3  / precision
    
    # calculate bounds of the system
    bounds = np.empty([2,2], dtype = float) 
    bounds[0,0] = np.min([ np.min(locs1[:,0]), np.min(locs2[:,0]), np.min(locs3[:,0]) ])
    bounds[0,1] = np.max([ np.max(locs1[:,0]), np.max(locs2[:,0]), np.max(locs3[:,0]) ])
    bounds[1,0] = np.min([ np.min(locs1[:,1]), np.min(locs2[:,1]), np.min(locs3[:,1]) ])
    bounds[1,1] = np.max([ np.max(locs1[:,1]), np.max(locs2[:,1]), np.max(locs3[:,1]) ])
    
    # generating the matrices to be plotted
    channel1 = generate_matrix(locs1, bounds)
    channel2 = generate_matrix(locs2, bounds)
    channel3 = generate_matrix(locs3, bounds)
    
    return channel1, channel2, channel3, bounds


def generate_matrix(locs , bounds, error=None):
    '''
    Takes the localizations and puts them in a matrix
    Parameters
    ----------
    img_param : Image()
        Class containing the data of the image.
    locs : Nx2 matrix float
        The actual locations of the localizations.
    bounds : 2x2 matrix 
        containing the bounds of all three systems
    error : Nx2 matrix, optional
        If not None, it will generate the matrix with the error per localization given as height
        
    Returns
    -------
    channel : matrix
        Contains an image of the localizations.
    '''
    size_img = np.round( (bounds[:,1] - bounds[:,0]) , 0).astype('int')

    channel = np.zeros([size_img[0], size_img[1]], dtype = int)
    if error is not None: counter = np.zeros([size_img[0], size_img[1]], dtype = int)
    for i in range(locs.shape[0]):
        loc = np.round(locs[i,:],0).astype('int')
        if isin_domain(loc, bounds):
            loc -= np.round(bounds[:,0],0).astype('int') # place the zero point on the left
            if error is None:
                channel[loc[0]-1, loc[1]-1] = 1
            else:
                channel[loc[0]-1, loc[1]-1] += error[i]
                counter[loc[0]-1, loc[1]-1] +=1           # normalization to get average error per cell
    if error is not None:
        channel = np.where(channel==0, 'nan')
        counter = np.where(counter==0, 'nan')
        channel = channel/counter
    return channel


def isin_domain(pos, img):
    '''
    checks if pos is within img domain

    Parameters
    ----------
    pos : 2 float array
        Position of a certain point.
    img : 2x2 float array
        The bounds of a certain image.

    Returns
    -------
    Bool
        True if pos is in img, False if not.

    '''
    return ( pos[0] > img[0,0] and pos[1] > img[1,0] and 
            pos[0] < img[0,1] and pos[1] < img[1,1] )


def bounds_check(locs1, locs3, max_deform = 150):
    '''
    Checks if mapping function is within limits 

    Parameters
    ----------
    locs1 ,locs3 : Nx2 float array
        Contains the localizations in pix.
    max_deform : float, optional
        The maximum amount of deformation allowed in nm. The default is 150.

    Returns
    -------
    None.

    '''
    max_loc3 = np.round(np.max([ np.max(locs3[:,0]), np.max(locs3[:,1]), 
                       np.abs(np.min(locs3[:,0])), np.abs(np.min(locs3[:,1]))
                       ]),0)
    max_loc1 = np.round(np.max([ np.max(locs1[:,0]), np.max(locs1[:,1]), 
                       np.abs(np.min(locs1[:,0])), np.abs(np.min(locs1[:,1]))
                       ]),0)
    if max_loc3 > 2*max_loc1 or max_loc3 < 0.5*max_loc1:
        print('\nI: Mapping might exceeds the threshold of', max_deform,'nm')
        print('(channel_1 max = ',max_loc1,', channel_2 max = ',max_loc3, 'nm)\n')
        input('Press ENTER if you want to continue:')
    elif max_loc3 > max_loc1 + max_deform or max_loc3 < max_loc1 - max_deform:
        print('\nI: Mapping might exceeds the threshold of', max_deform,'nm')
        print('(channel_1 max = ',max_loc1,', channel_2 max = ',max_loc3, 'nm)\n')
        time.sleep(2)