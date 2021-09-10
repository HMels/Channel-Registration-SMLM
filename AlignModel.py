# -*- coding: utf-8 -*-
"""
The align class
"""
import numpy as np
from photonpy import Dataset
import tensorflow as tf
import copy
import matplotlib.pyplot as plt


from Align_Modules.Affine import AffineModel
from Align_Modules.Polynomial3 import Polynomial3Model
from Align_Modules.RigidBody import RigidBodyModel
from Align_Modules.Splines import SplinesModel


#%% Align class
class AlignModel:
    def __init__(self, path, subset=None, align_rcc=True, coupled=False):
        '''
        Very simplistic version of something that should look like the Database class

        Parameters
        ----------
        path : List
            List containing one or two path locations.

        Returns
        -------
        None.

        '''
        self.shift_rcc=None
        self.subset=subset
        self.coupled=coupled
        self.gridsize=None
        
        
        ## Loading dataset
        if len(path)==1:
            # Dataset is grouped, meaning it has to be split manually
            print('Loading dataset... \n Grouping...')
            ds = Dataset.load(path[0],saveGroups=True)
            self.ch1 = ds[ds.group==0]
            self.ch2 = ds[ds.group==1]
        elif len(path)==2:
            # Dataset consists over 2 files
            print('Loading dataset...')
            self.ch1 = Dataset.load(path[0])
            self.ch2 = Dataset.load(path[1])
        else:
            raise TypeError('Path invalid, should be List of size 1 or 2.')
        
        
        self.ch2_original=copy.deepcopy(self.ch2)                               # making a copy of the original channel
        self.img, self.imgsize, self.mid = self.imgparams()                     # loading the image parameters
        if align_rcc: self.align_rcc()                                          # pre-aligning datasets via rcc
        if self.subset is not None and self.subset!=1:                          # loading a subset of data
            print('Loading subset of', self.subset)
            self.ch1, self.ch2 = self.load_subset(self.subset)
            
            
          
    #%% Loading the dataset functions
    def load_subset(self, subset):
    # loading subset of dataset
        l_grid = self.mid - np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2
        r_grid = self.mid + np.array([ subset*self.imgsize[0], subset*self.imgsize[1] ])/2
        
        indx1 = np.argwhere( (self.ch1.pos[:,0] >= l_grid[0]) * (self.ch1.pos[:,1] >= l_grid[1])
                            * (self.ch1.pos[:,0] <= r_grid[0]) * (self.ch1.pos[:,1] <= r_grid[1]) )[:,0]
        
        indx2 = np.argwhere( (self.ch2.pos[:,0] >= l_grid[0]) * (self.ch2.pos[:,1] >= l_grid[1])
                            * (self.ch2.pos[:,0] <= r_grid[0]) * (self.ch2.pos[:,1] <= r_grid[1]) )[:,0]
    
        return self.ch1.pos[ indx1, : ], self.ch2.pos[ indx2, : ]
    
            
    def imgparams(self):
    # calculate borders of system
    # returns a 2x2 matrix containing the edges of the image, a 2-vector containing
    # the size of the image and a 2-vector containing the middle of the image
        img = np.empty([2,2], dtype = float)
        img[0,0] = np.min(( np.min(self.ch1.pos[:,0]), np.min(self.ch2.pos[:,0]) ))
        img[1,0] = np.max(( np.max(self.ch1.pos[:,0]), np.max(self.ch2.pos[:,0]) ))
        img[0,1] = np.min(( np.min(self.ch1.pos[:,1]), np.min(self.ch2.pos[:,1]) ))
        img[1,1] = np.max(( np.max(self.ch1.pos[:,1]), np.max(self.ch2.pos[:,1]) ))
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
    
    
    def align_rcc(self):
    # align the dataset using a RCC shift
        print('Alignning both datasets')
        self.shift_rcc = Dataset.align(self.ch1, self.ch2)
        print('\nRCC shift equals', self.shift_rcc)
        if not np.isnan(self.shift_rcc).any():
            self.ch1.pos+= self.shift_rcc
        else: 
            print('Warning: RCC Shift undefined and will be skipped')
            
            
    def couple_dataset(self, maxDist=50, Filter=True):
    # couples dataset with a simple iterative nearest neighbour method
        print('Coupling datasets with an iterative method...')
        locsA=[]
        locsB=[]
        for i in range(self.ch1.pos.shape[0]):
            dists = np.sqrt(np.sum((self.ch1.pos[i,:]-self.ch2.pos)**2,1))
            if not Filter or np.min(dists)<maxDist:
                locsA.append( self.ch1.pos[i,:] )
                locsB.append( self.ch2.pos[np.argmin(dists),:] ) 
        
        # initialize the new coupled dataset
        self.ch1.pos = np.array(locsA)
        self.ch2.pos = np.array(locsB)
        self.coupled = True
    
    
    #%% Optimization functions
    '''
    atm everything is only for coupled datasets, so no KNN 3dim tensors
    '''
    #@tf.function
    def train_model(self, model, Nit, opt, ch1_tf=None, ch2_tf=None):
    # The training loop of the model
        if ch1_tf is None: ch1_tf=tf.Variable(self.ch1.pos, dtype=tf.float32, trainable=False) 
        if ch2_tf is None: ch2_tf=tf.Variable(self.ch2.pos, dtype=tf.float32, trainable=False)
        
        for i in range(Nit):
            with tf.GradientTape() as tape:
                entropy = model(ch1_tf, ch2_tf)
            
            grads = tape.gradient(entropy, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
        return model
            
    
    #%% Global Transforms (Affine, Polynomial3, RigidBody)
    ## Affine
    def Train_Affine(self, lr=1, Nit=200):
    # Training the Affine Mapping
        # initializing the model and optimizer
        self.AffineModel=AffineModel(direct=self.coupled)
        opt=tf.optimizers.Adagrad(lr)
        
        # Training the Model
        print('Training Affine Mapping...')
        self.AffineModel = self.train_model(self.AffineModel, Nit, opt)
        
    
    def Transform_Affine(self):
    # Transforms ch2 according to the Model
        print('Transforming Affine Mapping...')
        ch2_tf=tf.Variable(self.ch2.pos, dtype=tf.float32, trainable=False)
        ch2_tf=self.AffineModel.transform_vec(ch2_tf)
        self.ch2.pos=np.array(ch2_tf.numpy())
      
        
    ## Polynomial3
    def Train_Polynomial3(self, lr=1, Nit=200):
    # Training the Polynomial3 Mapping
        # initializing the model and optimizer
        self.Polynomial3Model=Polynomial3Model(direct=self.coupled)
        opt=tf.optimizers.Adagrad(lr)
        
        # Training the Model
        print('Training Polynomial3 Mapping...')
        self.Polynomial3Model = self.train_model(self.Polynomial3Model, Nit, opt)
        
    
    def Transform_Polynomial3(self):
    # Transforms ch2 according to the Model
        print('Transforming Polynomial3 Mapping...')
        ch2_tf=tf.Variable(self.ch2.pos, dtype=tf.float32, trainable=False)
        ch2_tf=self.Polynomial3Model.transform_vec(ch2_tf)
        self.ch2.pos=np.array(ch2_tf.numpy())
        
    
    ## RigidBody
    def Train_RigidBody(self, lr=1, Nit=200):
    # Training the RigidBody Mapping
        # initializing the model and optimizer
        self.RigidBodyModel=RigidBodyModel(direct=self.coupled)
        opt=tf.optimizers.Adagrad(lr)
        
        # Training the Model
        print('Training RigidBody Mapping...')
        self.RigidBodyModel = self.train_model(self.RigidBodyModel, Nit, opt)
        
    
    def Transform_RigidBody(self):
    # Transforms ch2 according to the Model
        print('Transforming RigidBody Mapping')
        ch2_tf=tf.Variable(self.ch2.pos, dtype=tf.float32, trainable=False)
        ch2_tf=self.RigidBodyModel.transform_vec(ch2_tf)
        self.ch2.pos=np.array(ch2_tf.numpy())
        
     
    #%% CatmullRom Splines
    ## Splines
    def generate_grid(self, gridsize, sys_param=None):
    # Creates a grid for the Splines. The outer two gridpoints are used as anchor
        print('Initializing Spline Grid...')
        
        # normalize dataset and generate or import parameters
        ch2_input = tf.Variable(self.ch2.pos / gridsize)
        self.gridsize = gridsize
        if sys_param is None:
            x1_min = tf.reduce_min(tf.floor(ch2_input[:,0]))
            x2_min = tf.reduce_min(tf.floor(ch2_input[:,1]))
            x1_max = tf.reduce_max(tf.floor(ch2_input[:,0]))
            x2_max = tf.reduce_max(tf.floor(ch2_input[:,1]))
        else:
            x1_min = tf.floor(sys_param[0,0]/gridsize)
            x2_min = tf.floor(sys_param[0,1]/gridsize)
            x1_max = tf.floor(sys_param[1,0]/gridsize)
            x2_max = tf.floor(sys_param[1,1]/gridsize)
        
        # generating the grid
        x1_grid = tf.range(x1_min-2, x1_max+4, 1)
        x2_grid = tf.range(x2_min-2, x2_max+4, 1)
        self.CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
        
        # initializing the indexes of ch2
        self.CP_idx = tf.cast(tf.stack(
            [( ch2_input[:,0]-x1_min+2)//1 , ( ch2_input[:,1]-x2_min+2)//1 ]
            , axis=1), dtype=tf.int32)
        
    
    def Train_Splines(self, lr=1, Nit=200, gridsize=1000):
    # Training the Splines Mapping
        # initializing the model and optimizer
        self.generate_grid(gridsize)
        self.SplinesModel=SplinesModel(self.CP_locs, self.CP_idx, direct=self.coupled)
        opt=tf.optimizers.Adagrad(lr)
        
        # Training the Model
        print('Training Splines Mapping...')
        self.SplinesModel = self.train_model(
            self.SplinesModel, Nit, opt, 
            ch1_tf=tf.Variable(self.ch1.pos/gridsize, trainable=False),
            ch2_tf=tf.Variable(self.ch2.pos/gridsize, trainable=False)
            )
        
    
    def Transform_Splines(self, sys_param=None):
    # Transforms ch2 according to the Model
        print('Transforming Splines Mapping')
        if self.gridsize is None: raise Exception('No Grid has been generated yet')
        
        # normalize dataset and generate or import parameters
        ch2_tf=tf.Variable(self.ch2.pos/self.gridsize, dtype=tf.float32, trainable=False)
        if sys_param is None:
            x1_min = tf.reduce_min(tf.floor(ch2_tf[:,0]))
            x2_min = tf.reduce_min(tf.floor(ch2_tf[:,1]))
        else:
            x1_min = tf.floor(sys_param[0,0]/self.gridsize)
            x2_min = tf.floor(sys_param[0,1]/self.gridsize)
            
        # The spline indexes of the new ch2
        CP_idx = tf.cast(tf.stack(
            [( ch2_tf[:,0]-x1_min+2)//1 , ( ch2_tf[:,1]-x2_min+2)//1 ], 
            axis=1), dtype=tf.int32)
        
        # initialize this new ch2 model
        SplinesModel_temp = copy.copy( self.SplinesModel )
        SplinesModel_temp.reset_CP(CP_idx)
            
        # transform the new ch2 model
        ch2_tf=self.SplinesModel.transform_vec(ch2_tf) * self.gridsize
        self.ch2.pos=np.array(ch2_tf.numpy())
        
        
    ## Plotting the Grid
    def plot_SplineGrid(self, ch1=None, ch2=None, ch2_original=None, d_grid=.1, lines_per_CP=1, 
                        locs_markersize=10, CP_markersize=8, grid_markersize=3, grid_opacity=1,
                        sys_param=None): 
        '''
        Plots the grid and the shape of the grid in between the Control Points
    
        Parameters
        ----------
        ch1 , ch2 , ch2_original : Nx2 tf.float32 tensor
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
        if ch1 is None:
            ch1=self.ch1.pos
            ch2=self.ch2.pos
            ch2_original=self.ch2_original.pos
            
        
        if sys_param is None:
            x1_min = tf.reduce_min(tf.floor(ch2_original[:,0]/self.gridsize))
            x2_min = tf.reduce_min(tf.floor(ch2_original[:,1]/self.gridsize))
            x1_max = tf.reduce_max(tf.floor(ch2_original[:,0]/self.gridsize))
            x2_max = tf.reduce_max(tf.floor(ch2_original[:,1]/self.gridsize))
        else:
            x1_min = tf.floor(sys_param[0,0]/self.gridsize)
            x2_min = tf.floor(sys_param[0,1]/self.gridsize)
            x1_max = tf.floor(sys_param[1,0]/self.gridsize)
            x2_max = tf.floor(sys_param[1,1]/self.gridsize)
        
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
        SplinesModel_temp = copy.copy(self.SplinesModel)
        SplinesModel_temp.reset_CP(CP_idx)
        grid_tf = SplinesModel_temp.transform_vec(grid_tf)
        
        # plotting the localizations
        plt.figure()
        plt.plot(ch2[:,0],ch2[:,1], color='red', marker='.', linestyle='',
                 markersize=locs_markersize, label='Mapped CH2')
        plt.plot(ch2_original[:,0],ch2_original[:,1], color='orange', marker='.', linestyle='', 
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
            plt.plot(grid_tf1[:,0]*self.gridsize,grid_tf1[:,1]*self.gridsize, 'b.',
                     markersize=grid_markersize, alpha=grid_opacity)
            plt.plot( self.CP_locs[:,:,0]*self.gridsize,  self.CP_locs[:,:,1]*self.gridsize, 
                     'b+', markersize=CP_markersize)
    
            # plotting the inbetween gridlines
            plt.plot(grid_tf2[:,0]*self.gridsize,grid_tf2[:,1]*self.gridsize, 'c.',
                     markersize=grid_markersize, alpha=grid_opacity)
        
        else:
            # plotting the gridlines
            plt.plot(grid_tf[:,0]*self.gridsize,grid_tf[:,1]*self.gridsize, 'b.',
                     markersize=grid_markersize, alpha=grid_opacity)
            plt.plot( self.CP_locs[:,:,0]*self.gridsize,  self.CP_locs[:,:,1]*self.gridsize, 
                     'b+', markersize=CP_markersize)
            
        plt.legend()
    