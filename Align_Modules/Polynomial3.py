# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:07:41 2021

@author: Mels
"""
import tensorflow as tf

class Polynomial3Model(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the pts, the [x1,x2] locations of all localizations
    - gives it a certain polynomial deformation via the Polynomial Class
    - calculates the relative entropy via Rel_entropy()    
    '''
    
    def __init__(self, direct=False, name = 'polynomial'): 
        super().__init__(name=name)
        self.M1 = tf.Variable([[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              dtype=tf.float32, trainable=True, name = 'M1'
                              )
        self.M2 = tf.Variable([[0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              dtype=tf.float32, trainable=True, name = 'M2'
                              )
    
    
    @tf.function 
    def call(self, pts):
        if len(pts.shape)==2: # transform vectors
            y = tf.stack([
                tf.concat([self.M1[0,0]*tf.ones([pts.shape[0],1]), 
                           self.M1[1,0]*pts[:,0][:,None],
                           self.M1[0,1]*pts[:,1][:,None],
                           self.M1[1,1]*(pts[:,0]*pts[:,1])[:,None],
                           self.M1[2,1]*((pts[:,0]**2)*pts[:,1])[:,None],
                           self.M1[2,2]*((pts[:,0]*pts[:,1])**2)[:,None],
                           self.M1[1,2]*(pts[:,0]*(pts[:,1]**2))[:,None],
                           self.M1[0,2]*(pts[:,1]**2)[:,None],
                           self.M1[2,0]*(pts[:,0]**2)[:,None]
                           ], axis = 1),
                tf.concat([self.M2[0,0]*tf.ones([pts.shape[0],1]), 
                           self.M2[1,0]*pts[:,0][:,None],
                           self.M2[0,1]*pts[:,1][:,None],
                           self.M2[1,1]*(pts[:,0]*pts[:,1])[:,None],
                           self.M2[2,1]*((pts[:,0]**2)*pts[:,1])[:,None],
                           self.M2[2,2]*((pts[:,0]*pts[:,1])**2)[:,None],
                           self.M2[1,2]*(pts[:,0]*(pts[:,1]**2))[:,None],
                           self.M2[0,2]*(pts[:,1]**2)[:,None],
                           self.M2[2,0]*(pts[:,0]**2)[:,None]
                           ], axis = 1),
                ], axis = 2)
            return tf.reduce_sum(y, axis = 1)
        
        elif len(pts.shape)==3: # transform matrices
            y = tf.stack([
                tf.concat([self.M1[0,0]*tf.ones([1, pts.shape[0], pts.shape[1]]), 
                           self.M1[1,0]*pts[:,:,0][None],
                           self.M1[0,1]*pts[:,:,1][None],
                           self.M1[1,1]*(pts[:,:,0]*pts[:,:,1])[None],
                           self.M1[2,1]*((pts[:,:,0]**2)*pts[:,:,1])[None],
                           self.M1[2,2]*((pts[:,:,0]*pts[:,:,1])**2)[None],
                           self.M1[1,2]*(pts[:,:,0]*(pts[:,:,1]**2))[None],
                           self.M1[0,2]*(pts[:,:,1]**2)[None],
                           self.M1[2,0]*(pts[:,:,0]**2)[None]
                           ], axis = 0)[:,:,:,None],
                tf.concat([self.M2[0,0]*tf.ones([1, pts.shape[0], pts.shape[1]]), 
                           self.M2[1,0]*pts[:,:,0][None],
                           self.M2[0,1]*pts[:,:,1][None],
                           self.M2[1,1]*(pts[:,:,0]*pts[:,:,1])[None],
                           self.M2[2,1]*((pts[:,:,0]**2)*pts[:,:,1])[None],
                           self.M2[2,2]*((pts[:,:,0]*pts[:,:,1])**2)[None],
                           self.M2[1,2]*(pts[:,:,0]*(pts[:,:,1]**2))[None],
                           self.M2[0,2]*(pts[:,:,1]**2)[None],
                           self.M2[2,0]*(pts[:,:,0]**2)[None]
                           ], axis = 0)[:,:,:,None]
                ], axis = 3)
            return tf.reduce_sum(tf.reduce_sum(y, axis = 0), axis = 3)
        
        else: ValueError('Invalid input shape! ch1 has shape '+str(pts.shape) )