# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:18:47 2021

@author: Mels
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import time

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform

def calculate_curve(pos1, pos2, xlim):
    def func(r, sigma):
        # from Churchman et al 2006
        sigma2=sigma**2
        return r/sigma2*np.exp(-r**2/2/sigma2)
    
    dist = np.sqrt( np.sum( ( pos1 - pos2 )**2, axis = 1) )
    dist=dist[np.argwhere(dist<xlim)]
    n=np.histogram(dist)
    
    N = pos1.shape[0] * ( n[1][1]-n[1][0] )
    xn=(n[1][:-1]+n[1][1:])/2
    popt, pcov = curve_fit(func, xn, n[0]/N, p0=np.std(xn))
        
        
if True: #% Load Excel Niekamp
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True, FrameOptimization=False)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.pix_size=159
    DS2.pix_size=DS1.pix_size
    DS1.link_dataset()
    DS2.link_dataset()
    
    ## optimization params
    learning_rates = [1e3, .1, 1e-3]
    epochs = [100, None, 200]
    pair_filter = [250, 30, 30]
    gridsize=3000