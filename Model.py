# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:16:12 2021

@author: Mels
"""
import copy
import pandas as pd
import tensorflow as tf
import numpy as np
from photonpy import PostProcessMethods, Context, Dataset
import time
from scipy.optimize import curve_fit

from Registration import Registration

from Align_Modules.Affine import AffineModel
from Align_Modules.Polynomial3 import Polynomial3Model
from Align_Modules.RigidBody import RigidBodyModel
from Align_Modules.Splines import CatmullRomSpline2D
from Align_Modules.Shift import ShiftModel


class Model(Registration):
    def __init__(self, execute_linked=True):     
        ## Models
        self.AffineModel = None
        self.Polynomial3Model = None
        self.RigidBodyModel = None
        self.ShiftModel = None
        self.SplinesModel = None
        self.CP_locs = None
        self.gridsize=None
        self.edge_grids=None
        
        ## Neighbours
        self.ControlPoints=None
        self.NN_maxDistance=None
        self.NN_threshold=None
        self.Neighbours=False   
        
        self.execute_linked=execute_linked
        Registration.__init__(self, execute_linked=execute_linked)
        
        
    #%% model
    def TrainRegistration(self, execute_linked=None, learning_rates=[1e3,.1,1e-3], epochs=[100,100,100], pair_filter=[250,30],
                         gridsize=3000, edge_grids=1):
        if execute_linked is not None: self.execute_linked=execute_linked
        else: self.execute_linked=self.linked
        
        start=time.time()
        
        #% Shift Transform
        self.ShiftModel=ShiftModel()
        self.Train_Model(self.ShiftModel, lr=learning_rates[0], epochs=epochs[0], opt_fn=tf.optimizers.Adagrad)
        self.Transform_Model(self.ShiftModel)
        
        if pair_filter[0] is not None:
            self.Filter(pair_filter[0]) 
            
        #% Affine Transform
        self.AffineModel=AffineModel()
        self.Train_Model(self.AffineModel, lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
        self.Transform_Model(self.AffineModel)
            
        #% CatmullRomSplines
        if epochs[2] is not None:
            self.execute_linked=True #%% Splines can only be optimized by pair-optimization
            if not self.linked: self.link_dataset(maxDistance=pair_filter[2])
            
            # initializing and training the model
            ch1_input,ch2_input=self.InitializeSplines(gridsize=gridsize, edge_grids=edge_grids)
            self.SplinesModel=CatmullRomSpline2D(self.ControlPoints)
            self.Train_Model(self.SplinesModel, lr=learning_rates[2], epochs=epochs[2], opt_fn=tf.optimizers.SGD, 
                             ch1=ch1_input, ch2=ch2_input)  
            
            # applying the model
            self.ControlPoints = self.SplinesModel.ControlPoints
            self.ch2.pos.assign(self.InputSplines(
                self.Transform_Model(self.SplinesModel, ch2=self.InputSplines(self.ch2.pos)),
                inverse=True))
            if self.Neighbours:
                self.ch2NN.pos.assign(self.InputSplines(
                    self.Transform_Model(self.SplinesModel, ch2=self.InputSplines(self.ch2NN.pos)),
                    inverse=True))
        #self.PlotSplineGrid(plotarrows=False)
            
        if pair_filter[1] is not None:
            self.Filter(pair_filter[1])
        print('Optimized in',round((time.time()-start)/60,1),'minutes!')
        
        
    def ApplyRegistration(self):
            if self.ShiftModel is not None: self.Transform_Model(self.ShiftModel)
            if self.AffineModel is not None: self.Transform_Model(self.AffineModel)
            if self.RigidBodyModel is not None: self.Transform_Model(self.RigidBodyModel)
            if self.Polynomial3Model is not None: self.Transform_Model(self.Polynomial3Model)
            if self.SplinesModel is not None: self.ch2.pos.assign(self.InputSplines(
                    self.Transform_Model(self.SplinesModel, ch2=self.InputSplines(self.ch2.pos)),
                    inverse=True))
        
        
    #%% SimpleShift
    def SimpleShift(self, other=None, maxDistance=2000):
        print('Shifting the channels according to a simple nearest neighbour least squares optimisation...')
        start=time.time()
        linked=self.linked
        if not self.linked: self.link_dataset(maxDistance=maxDistance)
        
        distx=self.ch1.pos[:,0]-self.ch2.pos[:,0]
        disty=self.ch1.pos[:,1]-self.ch2.pos[:,1]
        nx=np.histogram(distx)
        ny=np.histogram(disty)
        
        ## fit bar plot data using curve_fit
        def func(r, mu, sigma):
            return np.exp(-(r - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2*np.pi)*sigma)
        
        Nx = self.ch1.pos.shape[0] * ( nx[1][1]-nx[1][0] )
        Ny = self.ch1.pos.shape[0] * ( ny[1][1]-ny[1][0] )
        xn=(nx[1][:-1]+nx[1][1:])/2
        yn=(ny[1][:-1]+ny[1][1:])/2
        poptx, pcovx = curve_fit(func, xn, nx[0]/Nx, p0=[np.average(distx), np.std(distx)])
        popty, pcovy = curve_fit(func, yn, ny[0]/Ny, p0=[np.average(disty), np.std(disty)])
        
        if not linked: self.reload_dataset()
        print('Optimized SimpleShift in',round((time.time()-start)/60,1),'minutes!')
        self.ch2.pos.assign(
            tf.stack([self.ch2.pos[:,0]+poptx[0],self.ch2.pos[:,1]+popty[0]],axis=1)
            )
        if other is not None:
            other.ch2.pos.assign(
                tf.stack([other.ch2.pos[:,0]+poptx[0],other.ch2.pos[:,1]+popty[0]],axis=1)
                )
            return other
        
        
    #%% miscaleneous fn
    def copy_models(self, other):
        self.AffineModel = copy.deepcopy(other.AffineModel)
        self.Polynomial3Model = copy.deepcopy(other.Polynomial3Model)
        self.RigidBodyModel = copy.deepcopy(other.RigidBodyModel)
        self.ShiftModel = copy.deepcopy(other.ShiftModel)
        self.SplinesModel = copy.deepcopy(other.SplinesModel)
        
        self.CP_locs = copy.deepcopy(other.CP_locs)
        self.gridsize = other.gridsize
        self.edge_grids = other.edge_grids
        if self.gridsize is not None:
            self.x1_min = other.x1_min
            self.x2_min = other.x2_min
            self.x1_max = other.x1_max
            self.x2_max = other.x2_max