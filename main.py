# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt

from Align_Datasets.Dataset_hdf5 import Dataset_hdf5
from Align_Datasets.Dataset_excel import Dataset_excel
from Align_Datasets.Generate_Dataset import Generate_Dataset

plt.close('all')

#%% Load datasets
if False: #% Load Beads
    DS1 = Dataset_hdf5(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
               align_rcc=False, coupled=True, pix_size=1)
    DS1, DS2 = DS1.SplitDataset()
    gridsize=200


if False: #% Load Clusters
    DS1 = Dataset_hdf5([ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
                        'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ],
                       align_rcc=False, coupled=False, pix_size=159)
    DS1.SubsetRandom(subset=0.2)
    DS1.couple_dataset(FrameLinking=False)
    DS1, DS2 = DS1.SplitDataset()
    gridsize=1000
    

if True: #% Load Excel
    DS1 = Dataset_excel('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                        align_rcc=False, coupled=False)
    DS2 = Dataset_excel('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                        align_rcc=False, coupled=False, pix_size=1)
    DS1.couple_dataset(FrameLinking=True)
    DS2.couple_dataset(FrameLinking=True)
    gridsize=3000


if False: #% Simulate Dataset beads
    DS1 = Generate_Dataset(coupled=True, imgshape=[512*159, 512*159], random_deform=(True))
    DS1.generate_dataset_beads(N=216, error=1, noise=0.005)
    DS1, DS2 = DS1.SplitDataset()
    gridsize=200
    
    
if False: #% Simulate Dataset clusters
    DS1 = Generate_Dataset(coupled=False, imgshape=[512*159, 512*159], random_deform=(True))
    DS1.generate_dataset_clusters(Nclust=100, N_per_clust=250, std_clust=7, error=10, noise=0.005)
    DS1, DS2 = DS1.SplitDataset()
    gridsize=1000

#%% Params
pair_filter = [250, 30]
DS1.developer_mode = False

#%% generate NN
if not DS1.coupled: 
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
#DS1.Train_Affine(lr=1, Nit=500)
#DS1.Transform_Affine()


#%% CatmullRomSplines
DS1.Train_Splines(lr=1e-2, Nit=100, gridsize=gridsize, edge_grids=1)
DS1.Transform_Splines()
#DS1.plot_SplineGrid()
DS1.Filter_Pairs(pair_filter[1])

#%% Mapping DS2 (either a second dataset or the cross validation)
if not DS1.developer_mode:
    ## Copy all mapping parameters
    DS2.copy_models(DS1)
    
    ## Shift and Affine transform
    DS2.Transform_Shift()
    #DS2.Transform_Affine()
    
    ## Splines transform
    DS2.reload_splines()
    DS2.Transform_Splines()
    DS2.Filter_Pairs(pair_filter[1])
    
    
    #%% output
    nbins=100
    xlim=pair_filter[1]
    
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
        
