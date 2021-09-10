# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import numpy as np

from AlignModel import AlignModel
from Analysis import Analysis


plt.close('all')

#%% Optimization
DS1 =AlignModel(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
           align_rcc=False, subset=1, coupled=True)

DS1.Train_Affine(lr=1, Nit=500)
DS1.Transform_Affine()

#%% CatmullRomSplines
DS1.Train_Splines(lr=1e-2, Nit=1000, gridsize=200)
DS1.Transform_Splines()
DS1.plot_SplineGrid()

#%% Plotting Error Distribution
An1=Analysis(AlignModel=(DS1))
An1.ErrorPlot()

#%% Plotting Channels
An1.generate_channel(precision=10)
An1.plot_channel()
An1.plot_1channel()