# generate_neighbours.py
"""
Created on Thu Apr 29 14:05:40 2021

@author: Mels
"""
import numpy as np
from photonpy import PostProcessMethods, Context


def find_all_neighbours(locs_A, locs_B, maxDistance = 50, k = 16):
    '''
    generates a list with arrays containing the neighbours via find_channel_neighbours
    It then deletes all none bright spots.  Also used to make sure output matrix has
    uniform size

    Parameters
    ----------
    locs_A, locs_B : 2xN float numpy array
        The locations of the localizations.
    threshold : int, optional
        The threshold of neighbouring locs needed to not be filtered. The default is None,
        which means the program will use a threshold of average + std
    maxDistance : float/int, optional
        The vicinity in which we search for neighbours. The default is 50.
    k : int
        The number of KNNs to generate if no brightest neighbours are found. the default is 32.
        
    Returns
    -------
    idxlist_new : list
        list filtered on bright spots, with size [2 x threshold]
        containing per indice of ch1 the neighbours in ch2.
    '''
    idxlist = find_channel_neighbours(locs_A, locs_B, maxDistance)

    print('Generating all neighbouring spots...')
    idx1list = []
    idx2list = []
    maxFill = 0
    for idx in idxlist:
        Fillsize = idx.size
        if Fillsize > 0:
            idx1list.append(idx[0,:]) 
            idx2list.append(idx[1,:]) 
            if Fillsize > maxFill: maxFill = Fillsize
    
    if idx1list == []: # generate the neighbours via kNN 
        print('\nError: No neighbours generated. Might be related to Threshold!',
              '\nNeighbours will be generated via KNN...')
        for i in range(locs_A.shape[0]):
            idx1list.append( (i * np.ones([k,1], dtype=int)) )
        idx2list = KNN(locs_A, locs_B, k)
        
    neighbours_A = generate_neighbour_matrix(idx1list, locs_A, maxFill)
    neighbours_B = generate_neighbour_matrix(idx2list, locs_B, maxFill)
    print('Found',neighbours_A.shape[1],'neighbours for',neighbours_A.shape[0],'localizations')
    return neighbours_A, neighbours_B


def find_bright_neighbours(locs_A, locs_B, threshold = None, maxDistance = 50, k = 16):
    '''
    generates a list with arrays containing the neighbours via find_channel_neighbours
    It then deletes all none bright spots.  Also used to make sure output matrix has
    uniform size

    Parameters
    ----------
    locs_A, locs_B : 2xN float numpy array
        The locations of the localizations.
    threshold : int, optional
        The threshold of neighbouring locs needed to not be filtered. The default is None,
        which means the program will use a threshold of average + std
    maxDistance : float/int, optional
        The vicinity in which we search for neighbours. The default is 50.
    k : int
        The number of KNNs to generate if no brightest neighbours are found. the default is 32.
        
    Returns
    -------
    idxlist_new : list
        list filtered on bright spots, with size [N x threshold x 2]
        containing per indice of ch1 the neighbours in ch2.
    '''
    idxlist = find_channel_neighbours(locs_A, locs_B, maxDistance)

    if threshold == None: # threshold = avg + std
        num = []
        for idx in idxlist:
            if idx.size>0:
                num.append(idx.shape[1])
        threshold = 20 #np.round(np.average(num) + np.std(num),0).astype('int')
    
    print('Filtering for brightest spots...')
    idx1list = []
    idx2list = []
    
    for idx in idxlist:
        if idx.size>0:
            if idx.shape[1] > threshold:
                # we want to have a max of threshold in our array
                idx1list.append(idx[0,
                                    np.random.choice(idx.shape[1], threshold)
                                    ]) 
                idx2list.append(idx[1,
                                    np.random.choice(idx.shape[1], threshold)
                                    ]) 
    
    idx1list = []
    if idx1list == []: # generate the neighbours via kNN 
        print('\nError: No neighbours generated. Might be related to Threshold!',
              '\nNeighbours will be generated via KNN...')
        for i in range(locs_A.shape[0]):
            idx1list.append( (i * np.ones(k, dtype=int)) )
        idx2list = KNN(locs_A, locs_B, k)
        
    neighbours_A = generate_neighbour_matrix(idx1list, locs_A)
    neighbours_B = generate_neighbour_matrix(idx2list, locs_B)
    print('Generated',neighbours_A.shape[1],'neighbours for',neighbours_A.shape[0],'localizations')
    return neighbours_A, neighbours_B


def find_channel_neighbours(locs_A, locs_B, maxDistance = 50):
    '''
    generates a list with arrays containing the neighbours

    Parameters
    ----------
    locs_A, locs_B : 2xN float numpy array
        The locations of the localizations.
    maxDistance : float/int, optional
        The vicinity in which we search for neighbours. The default is 50.

    Returns
    -------
    idxlist : list
        List containing per indice of ch1 the neighbours in ch2.

    '''
    print('Finding neighbours...')
    
    with Context() as ctx:
        counts,indices = PostProcessMethods(ctx).FindNeighbors(locs_A, locs_B, maxDistance)
    
    idxlist = []
    pos = 0
    i = 0
    for count in counts:
        idxlist.append( np.stack([
            i * np.ones([count], dtype=int),
            indices[pos:pos+count] 
            ]) )
        pos += count
        i += 1
            
    return idxlist


def generate_neighbour_matrix(idxlist, locs, maxFill = None):
    if maxFill == None:
        NN = []
        for nn in idxlist:
            NN.append(locs[nn,:])
    elif maxFill > 0:
        NN = []
        for nn in idxlist:
            fill = np.zeros( [maxFill - nn.size, 2] , dtype=float)
            NN.append(np.concatenate([ locs[nn,:] , fill ]))
    else: print('Error; Array Size [maxFill] invalid')
    return np.stack(NN)


def KNN(locs_A, locs_B, k):
    '''
    k-Nearest Neighbour Distance calculator

    Parameters
    ----------
    locs_A, locs_B : Nx2 float array
        The array containing the [x1, x2] locations of the localizations.
    k : int
        The number of kNN we want

    Returns
    -------
    knn : [k, N] TensorFlow Tensor
        Tensor Containing the matrix with the indices of k-nearest neighbours 

    '''
    knn = []
    for loc in locs_A:
        distances = np.sum((loc - locs_B)**2 , axis = 1)
        knn.append( np.argsort( distances )[:k] )
    
    return knn