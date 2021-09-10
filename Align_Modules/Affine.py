<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
The class containing the Affine Transform Optimization via TensorFlow
"""
import tensorflow as tf
from Align_Modules.Optimization_fn import Rel_entropy


class AffineModel(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    
    Parameters
    ----------
    d : tf.float32
        The amount of shift in nm
    A : 2x2 \Tensor tf.float32
        The Affine matrix
        
    Returns
    ----------
    Rel_entropy : tf.float32
        The relative entropy of the current mapping
        
    '''
    def __init__(self, direct=False, name='Affine'):
        super().__init__(name=name)
        self.direct=tf.Variable(direct, dtype=bool, trainable=False) # is the dataset coupled
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        self.A = tf.Variable([[1,0],[0,1]], dtype=tf.float32, trainable=True, name='A')
        

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
        
        x1 = ch2_mapped[:,:,0]*self.A[0,0] + ch2_mapped[:,:,1]*self.A[0,1]
        x2 = ch2_mapped[:,:,0]*self.A[1,0] + ch2_mapped[:,:,1]*self.A[1,1]
        return tf.stack([x1, x2], axis =2 )
    
    
    @tf.function
    def transform_vec(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None]
        
        x1 = ch2_mapped[:,0]*self.A[0,0] + ch2_mapped[:,1]*self.A[0,1]
        x2 = ch2_mapped[:,0]*self.A[1,0] + ch2_mapped[:,1]*self.A[1,1]
        return tf.stack([x1, x2], axis =1 )
    
    
=======
# -*- coding: utf-8 -*-
"""
The class containing the Affine Transform Optimization via TensorFlow
"""
import tensorflow as tf
from Align_Modules.Optimization_fn import Rel_entropy


class AffineModel(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    
    Parameters
    ----------
    d : tf.float32
        The amount of shift in nm
    A : 2x2 \Tensor tf.float32
        The Affine matrix
        
    Returns
    ----------
    Rel_entropy : tf.float32
        The relative entropy of the current mapping
        
    '''
    def __init__(self, direct=False, name='Affine'):
        super().__init__(name=name)
        self.direct=tf.Variable(direct, dtype=bool, trainable=False) # is the dataset coupled
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        self.A = tf.Variable([[1,0],[0,1]], dtype=tf.float32, trainable=True, name='A')
        

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
        
        x1 = ch2_mapped[:,:,0]*self.A[0,0] + ch2_mapped[:,:,1]*self.A[0,1]
        x2 = ch2_mapped[:,:,0]*self.A[1,0] + ch2_mapped[:,:,1]*self.A[1,1]
        return tf.stack([x1, x2], axis =2 )
    
    
    @tf.function
    def transform_vec(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None]
        
        x1 = ch2_mapped[:,0]*self.A[0,0] + ch2_mapped[:,1]*self.A[0,1]
        x2 = ch2_mapped[:,0]*self.A[1,0] + ch2_mapped[:,1]*self.A[1,1]
        return tf.stack([x1, x2], axis =1 )
    
    
>>>>>>> 620b2dc63c94701d67d9e60828504973bb6dae6a
    