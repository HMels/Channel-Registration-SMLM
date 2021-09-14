# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt

from Align_Datasets.Dataset_hdf5 import Dataset_hdf5
from Align_Datasets.Dataset_excel import Dataset_excel
from Align_Datasets.Generate_Dataset import Generate_Dataset
from Analysis import Analysis


plt.close('all')

pair_filter = [200, 30]  

#%% Load datasets
if False: #% Load Beads
    DS1 = Dataset_hdf5(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
               align_rcc=False, subset=1, coupled=True)
    DS1, DS2 = DS1.SplitDataset()


if True: #% Load Excel
    DS1 = Dataset_excel('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                        align_rcc=False, coupled=False)
    DS2 = Dataset_excel('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                        align_rcc=False, coupled=False)


if False: #% Simulate Dataset beads
    DS1 = Generate_Dataset(coupled=True, imgshape=[512, 512], random_deform=(True))
    DS1.generate_dataset_beads(N=216, error=10, noise=0.005)
    DS1, DS2 = DS1.SplitDataset()
    
    
if False: #% Simulate Dataset clusters
    DS1 = Generate_Dataset(coupled=False, imgshape=[512, 512], random_deform=(True))
    DS1.generate_dataset_clusters(Nclust=650, N_per_clust=250, std_clust=7, error=10, noise=0.005)
    DS1, DS2 = DS1.SplitDataset()


#%% Shift Transform
DS1.Train_Shift(lr=100, Nit=500)
DS1.Transform_Shift()


#%% Affine Transform
DS1.Filter_Pairs(pair_filter[0])
DS1.Train_Affine(lr=10, Nit=500)
DS1.Transform_Affine()


#%% CatmullRomSplines
DS1.Train_Splines(lr=1e-2, Nit=1000, gridsize=1000, edge_grids=2)
DS1.Transform_Splines()
#DS1.plot_SplineGrid()


#%% Plotting Error Distribution
DS1.Filter_Pairs(pair_filter[1])
An1=Analysis(AlignModel=(DS1))
An1.ErrorPlot()
An1.ErrorDistribution(nbins=30)


#%% Plotting Channels
if True:
    An1.generate_channel(precision=100)
    #An1.plot_channel()
    An1.plot_1channel()
    
    
#%% Mapping DS2 (either a second dataset or the cross validation)
## Copy all mapping parameters
DS2.copy_models(DS1)

## Shift and Affine transform
DS2.Transform_Shift()
DS2.Filter_Pairs(pair_filter[0])
DS2.Transform_Affine()

## Splines transform
DS2.reload_splines()
DS2.Transform_Splines()


#%% output DS2
DS2.Filter_Pairs(pair_filter[1])
An2=Analysis(AlignModel=(DS2))
An2.ErrorPlot()
An2.ErrorDistribution(nbins=30)

An1.ErrorPlotImage(An2)