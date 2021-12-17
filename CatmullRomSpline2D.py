# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:23:12 2021

@author: Mels
"""
import tensorflow as tf
import numpy as np

class _CatmullRomSplineBase(tf.keras.Model):
    
    def __init__(self):
        super(_CatmullRomSplineBase, self).__init__()

        hermiteBasis = np.array([
             [2, -2, 1, 1],
             [-3, 3,-2,-1],
             [0,  0, 1, 0],
             [1,  0, 0, 0]])
        
        catmullRom = np.array([
            [0,1,0,0],
            [0,0,1,0],
            [-0.5,0,0.5,0],
            [0,-0.5,0,0.5]])
    
        A = (( hermiteBasis @ catmullRom )[::-1]).copy()
        self.spline_basis  = tf.Variable(A, trainable=False, dtype=tf.float32)
        
    def __str__(self):
        return f'Catmull Rom Spline with ControlPoints ({self.ControlPoints.shape})'


class CatmullRomSpline2D(_CatmullRomSplineBase):
    """
    Spline that maps a 2D coordinate onto a n-d output
    """
    def __init__(self, ControlPoints):
        super(CatmullRomSpline2D, self).__init__()
        
        assert(len(ControlPoints.shape)==3)
        self.ControlPoints = tf.Variable(ControlPoints, trainable=True, dtype=tf.float32)
        
        
    @tf.function 
    def call(self, pts):
        if len(pts.shape)==2: # transform vectors
            x = pts[:,0]
            y = pts[:,1]
    
            ix = tf.cast(x, tf.int32)
            sx = tf.clip_by_value(x-tf.floor(x), 0, 1)
    
            iy = tf.cast(y, tf.int32)
            sy = tf.clip_by_value(y-tf.floor(y), 0, 1)
    
            iy = tf.clip_by_value(((iy-1)[:,None] + tf.range(4)[None,:]), 0, self.ControlPoints.shape[0]-1)
            ix = tf.clip_by_value(((ix-1)[:,None] + tf.range(4)[None,:]), 0, self.ControlPoints.shape[1]-1)
    
            # compute sx^a * A
            cx = (sx[:,None]**(tf.range(4,dtype=tf.float32)[None])) @ self.spline_basis
            cy = (sy[:,None]**(tf.range(4,dtype=tf.float32)[None])) @ self.spline_basis
            
            ix = ix[:,None,:] * tf.ones(4,dtype=tf.int32)[None,:,None]
            iy = iy[:,:,None] * tf.ones(4,dtype=tf.int32)[None,None,:]
            #ix = tf.repeat(ix[:,None,:], 4, axis=1) # repeat x over y axis
            #iy = tf.repeat(iy[:,:,None], 4, axis=2) # repeat y over x axis
            idx = tf.stack([iy,ix],-1)
            sel_ControlPoints = tf.gather_nd(self.ControlPoints, idx)
            
            # sel_ControlPoints shape is [#evals, y-index, x-index, dims]
            return tf.reduce_sum((sel_ControlPoints * cy[:,:,None,None] * cx[:,None,:,None]), axis=(1,2))
        
        elif len(pts.shape)==3: # transform matrices
            x = pts[:,:,0]
            y = pts[:,:,1]
    
            ix = tf.cast(x, tf.int32)
            sx = tf.clip_by_value(x-tf.floor(x), 0, 1)
    
            iy = tf.cast(y, tf.int32)
            sy = tf.clip_by_value(y-tf.floor(y), 0, 1)
    
            iy = tf.clip_by_value(((iy-1)[:,:,None] + tf.range(4)[None,None,:]), 0, self.ControlPoints.shape[0]-1)
            ix = tf.clip_by_value(((ix-1)[:,:,None] + tf.range(4)[None,None,:]), 0, self.ControlPoints.shape[1]-1)
    
            # compute sx^a * A
            cx = (sx[:,:,None]**(tf.range(4,dtype=tf.float32)[None,None,:])) @ self.spline_basis
            cy = (sy[:,:,None]**(tf.range(4,dtype=tf.float32)[None,None,:])) @ self.spline_basis
            
            ix = ix[:,:,None,:] * tf.ones(4,dtype=tf.int32)[None,None,:,None]
            iy = iy[:,:,:,None] * tf.ones(4,dtype=tf.int32)[None,None,None,:]
            #ix = tf.repeat(ix[:,None,:], 4, axis=1) # repeat x over y axis
            #iy = tf.repeat(iy[:,:,None], 4, axis=2) # repeat y over x axis
            idx = tf.stack([iy,ix],-1)
            sel_ControlPoints = tf.gather_nd(self.ControlPoints, idx)
            
            # sel_ControlPoints shape is [#evals, y-index, x-index, dims]
            return tf.reduce_sum((sel_ControlPoints * cy[:,:,:,None,None] * cx[:,:,None,:,None]), axis=(2,3))
        
        else: raise ValueError('Invalid input shape! ch1 has shape '+str(pts.shape) )