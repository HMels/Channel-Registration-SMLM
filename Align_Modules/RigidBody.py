# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:08:06 2021

@author: Mels
"""
import tensorflow as tf

class RigidBodyModel(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    
    Parameters
    ----------
    d : tf.float32
        The amount of shift in nm
    cos : tf.float32
        The cosinus of the rotation angle
        
    Returns
    ----------
    Rel_entropy : tf.float32
        The relative entropy of the current mapping
        
    '''
    def __init__(self, name='RigidBody'):
        super().__init__(name=name)
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        self.cos = tf.Variable(1, dtype=tf.float32, trainable=True, name='rotation',
                               constraint=lambda t: tf.clip_by_value(t, -1, 1))
        

    @tf.function 
    def call(self, pts):
        if len(pts.shape)==2: # transform vectors
            ## Shift
            pts_mapped = pts + self.d[None]
            
            ## Rotate
            #sin = tf.sqrt(1-tf.pow(self.cos, 2))
            x1 = pts_mapped[:,0]*self.cos #- pts_mapped[:,1]*sin
            x2 = pts_mapped[:,1]*self.cos #+ pts_mapped[:,0]*sin 
            return tf.stack([x1, x2], axis =1 )
        
        elif len(pts.shape)==3: # transform matrices
            ## Shift
            pts_mapped = pts + self.d[None,None] 
            
            ## Rotate
            #sin = tf.sqrt(1-tf.pow(self.cos, 2))
            x1 = pts_mapped[:,:,0]*self.cos #- pts_mapped[:,:,1]*sin
            x2 = pts_mapped[:,:,1]*self.cos #+ pts_mapped[:,:,0]*sin
            return tf.stack([x1, x2], axis =2 )
        
        else: raise ValueError('Invalid input shape! ch1 has shape '+str(pts.shape) )