# -*- coding: utf-8 -*-
"""
The class containing the Affine Transform Optimization via TensorFlow
"""
import tensorflow as tf

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
    def __init__(self, name='Affine'):
        super().__init__(name=name)
        self.A = tf.Variable([[1,0],[0,1]], dtype=tf.float32, trainable=True, name='A')
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        

    @tf.function 
    def call(self, pts):
        if len(pts.shape)==2: # transform vectors
            ## Shift
            pts_mapped = pts + self.d[None]
            
            x1 = pts_mapped[:,0]*self.A[0,0] + pts_mapped[:,1]*self.A[0,1]
            x2 = pts_mapped[:,0]*self.A[1,0] + pts_mapped[:,1]*self.A[1,1]
            return tf.stack([x1, x2], axis =1 )
        
        elif len(pts.shape)==3: # transform matrices
            ## Shift
            pts_mapped = pts + self.d[None,None]
            
            x1 = pts_mapped[:,:,0]*self.A[0,0] + pts_mapped[:,:,1]*self.A[0,1]
            x2 = pts_mapped[:,:,0]*self.A[1,0] + pts_mapped[:,:,1]*self.A[1,1]
            return tf.stack([x1, x2], axis =2 )
        
        else: ValueError('Invalid input shape! ch1 has shape '+str(pts.shape) )