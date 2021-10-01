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
        self.direct=tf.Variable(direct, dtype=bool, trainable=False) # is the dataset coupled
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')


    @tf.function 
    def call(self, ch1, ch2):
        if self.direct:
            return self.transform_vec(ch2)
        else:
            return self.transform_mat(ch2)
    
    
    @tf.function
    def transform_mat(self, ch2):
        return ch2 + self.d[None,None] 
    
    
    @tf.function
    def transform_vec(self, ch2):
        return ch2 + self.d[None]