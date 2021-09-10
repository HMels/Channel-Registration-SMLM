<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:08:06 2021

@author: Mels
"""
import tensorflow as tf
from Align_Modules.Optimization_fn import Rel_entropy


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
    def __init__(self, direct=False, name='shift'):
        super().__init__(name=name)
        self.direct=tf.Variable(direct, dtype=bool, trainable=False) # is the dataset coupled
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        self.cos = tf.Variable(0, dtype=tf.float32, trainable=True, name='rotation')
        

    @tf.function 
    def call(self, ch1, ch2):
        if self.direct:
            ch2_mapped = self.transform_vec(ch2)
            return tf.reduce_sum(tf.square(ch1-ch2_mapped)) 
        else:
            # 
            ch2_mapped = self.transform_mat(ch2)
            return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.function
    def transform_mat(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None,None] 
        
        ## Rotate
        sin = tf.sqrt(1-tf.pow(self.cos, 2))
        x1 = ch2_mapped[:,:,0]*self.cos - ch2_mapped[:,:,1]*sin
        x2 = ch2_mapped[:,:,0]*sin + ch2_mapped[:,:,1]*self.cos
        return tf.stack([x1, x2], axis =2 )
    
    
    @tf.function
    def transform_vec(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None]
        
        ## Rotate
        sin = tf.sqrt(1-tf.pow(self.cos, 2))
        x1 = ch2_mapped[:,0]*self.cos - ch2_mapped[:,1]*sin
        x2 = ch2_mapped[:,0]*sin + ch2_mapped[:,1]*self.cos
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:08:06 2021

@author: Mels
"""
import tensorflow as tf
from Align_Modules.Optimization_fn import Rel_entropy


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
    def __init__(self, direct=False, name='shift'):
        super().__init__(name=name)
        self.direct=tf.Variable(direct, dtype=bool, trainable=False) # is the dataset coupled
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        self.cos = tf.Variable(0, dtype=tf.float32, trainable=True, name='rotation')
        

    @tf.function 
    def call(self, ch1, ch2):
        if self.direct:
            ch2_mapped = self.transform_vec(ch2)
            return tf.reduce_sum(tf.square(ch1-ch2_mapped)) 
        else:
            # 
            ch2_mapped = self.transform_mat(ch2)
            return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.function
    def transform_mat(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None,None] 
        
        ## Rotate
        sin = tf.sqrt(1-tf.pow(self.cos, 2))
        x1 = ch2_mapped[:,:,0]*self.cos - ch2_mapped[:,:,1]*sin
        x2 = ch2_mapped[:,:,0]*sin + ch2_mapped[:,:,1]*self.cos
        return tf.stack([x1, x2], axis =2 )
    
    
    @tf.function
    def transform_vec(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None]
        
        ## Rotate
        sin = tf.sqrt(1-tf.pow(self.cos, 2))
        x1 = ch2_mapped[:,0]*self.cos - ch2_mapped[:,1]*sin
        x2 = ch2_mapped[:,0]*sin + ch2_mapped[:,1]*self.cos
>>>>>>> 620b2dc63c94701d67d9e60828504973bb6dae6a
        return tf.stack([x1, x2], axis =1 )