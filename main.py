<<<<<<< HEAD
# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import numpy as np

from Align import Align
import OutputModules.output_fn as output_fn


plt.close('all')

#%% Optimization
DS1 =Align(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'], subset=1, coupled=True)

DS1.Train_Affine(lr=1, Nit=200)
DS1.Transform_Affine()

#%% CatmullRomSplines
DS1.Train_Splines(lr=1e-2, Nit=1000)
DS1.Transform_Splines()


#%% Plotting
nbins = 30                                          # Number of bins
avg1, avg2, fig1, _ = output_fn.errorHist(DS1.ch1.pos,  DS1.ch2_original.pos, DS1.ch2.pos,
                                          nbins=30, direct=True)
fig1.suptitle('Distribution of distances between neighbouring Localizations')
    

## FOV
_, _, fig2, _ = output_fn.errorFOV(DS1.ch1.pos,  DS1.ch2_original.pos, DS1.ch2.pos, direct=True)
fig2.suptitle('Distribution of error between neighbouring pairs over radius')
    
=======
# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import numpy as np

from Align import Align
import OutputModules.output_fn as output_fn


plt.close('all')

#%% Optimization
DS1 =Align(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'], subset=1, coupled=True)

DS1.Train_Affine(lr=1, Nit=200)
DS1.Transform_Affine()

#%% CatmullRomSplines
DS1.Train_Splines(lr=1e-2, Nit=1000)
DS1.Transform_Splines()


#%% Plotting
nbins = 30                                          # Number of bins
avg1, avg2, fig1, _ = output_fn.errorHist(DS1.ch1.pos,  DS1.ch2_original.pos, DS1.ch2.pos,
                                          nbins=30, direct=True)
fig1.suptitle('Distribution of distances between neighbouring Localizations')
    

## FOV
_, _, fig2, _ = output_fn.errorFOV(DS1.ch1.pos,  DS1.ch2_original.pos, DS1.ch2.pos, direct=True)
fig2.suptitle('Distribution of error between neighbouring pairs over radius')
    
>>>>>>> 620b2dc63c94701d67d9e60828504973bb6dae6a
print('\nI: The original average distance was', avg1,'. The mapping has', avg2)