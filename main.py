# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt

from Dataset import Dataset

plt.close('all')

#%% Load datasets
if False: #% Load Beads
    DS1 = Dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'], linked=True, pix_size=1)
    DS1.load_dataset_hdf5()
    DS1, DS2 = DS1.SplitDataset()
    gridsize=200


if False: #% Load Clusters
    DS1 = Dataset([ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
                        'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ], linked=False, pix_size=159)
    DS1.load_dataset_hdf5()
    DS1.SubsetRandom(subset=0.2)
    DS1, DS2 = DS1.SplitDataset()
    DS1.link_dataset(FrameLinking=False)
    gridsize=1000
    

if True: #% Load Excel
    DS1 = Dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv', linked=False, pix_size=1)
    DS2 = Dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv', linked=False, pix_size=1)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.link_dataset(FrameLinking=True)
    #DS1.SplitFrames()
    gridsize=3000


#%% Params
pair_filter = [500, 30]

#%% generate NN
if not DS1.linked: 
    maxDistance=250
    k=8
    DS1.find_neighbours(maxDistance, k)
    if DS1.NN_maxDist is not None:
        DS2.find_BrightNN(maxDistance, k)
    elif DS1.NN_k is not None:
        DS2.find_kNN(k)
    else: raise Exception('No neighbours generated for DS1')


#%% Shift Transform
DS1.Train_Shift(lr=100, Nit=100)
DS1.Transform_Shift()

#%% Affine Transform
DS1.Filter_Pairs(pair_filter[0])
DS1.Train_Affine(lr=1, Nit=500)
DS1.Transform_Affine()

#%% CatmullRomSplines
DS1.Train_Splines(lr=1e-2, Nit=100, gridsize=gridsize, edge_grids=1)
DS1.Transform_Splines()
#DS1.plot_SplineGrid()
#DS1.Filter_Pairs(pair_filter[1])


#%% Mapping DS2 (either a second dataset or the cross validation)
if not DS1.developer_mode:
    ## Copy all mapping parameters
    DS2.copy_models(DS1)
    
    ## Transforms
    DS2.Transform_Shift()
    DS2.Transform_Affine()
    DS2.Transform_Splines()
    
    
    #%% output
    nbins=100
    xlim=pair_filter[0]
    
    ## Coupling dataset
    DS2.link_dataset(FrameLinking=True)
    DS2.Filter_Pairs(pair_filter[0])
    
    ## Add all batches together for data processing
    #DS1.concat_batches()
    #DS2.concat_batches()
    
    ## DS1
    DS1.ErrorPlot(nbins=nbins)
    DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim)
    
    ## DS2
    DS2.ErrorPlot(nbins=nbins)
    DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim)
    
    ## DS1 vs DS2
    DS1.ErrorPlotImage(DS2)
    
    #%% image
    ## Image overview
    if False:
        DS1.generate_channel(precision=100)
        DS1.plot_channel()
        DS1.plot_1channel()