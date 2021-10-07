# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:08:06 2021

@author: Mels
"""
import tensorflow as tf

class ShiftModel(tf.keras.Model):
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
    def __init__(self, direct=False, name='shift'):
        super().__init__(name=name)
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')


    @tf.function 
    def call(self, pts):
        if len(pts.shape)==2: # transform vectors
            return pts + self.d[None]
        elif len(pts.shape)==3: # transform matrices
            return pts + self.d[None,None] 
        else: ValueError('Invalid input shape! ch1 has shape '+str(pts.shape) )