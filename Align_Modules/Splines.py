# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:23:12 2021

@author: Mels
"""
import tensorflow as tf
from Align_Modules.Optimization_fn import Rel_entropy

class SplinesModel(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, CP_locs, CP_idx, CP_idx_nn=None, direct=False, name='CatmullRomSplines'):
        super().__init__(name=name)
        self.direct=direct
        # The location of the ControlPoints. This will be trained
        self.CP_locs_trainable = tf.Variable(CP_locs[1:-1,1:-1,:], dtype=tf.float32,
                                   trainable=True, name='ControlPointstrainable')  
        self.CP_locs_untrainable_ax0 = tf.Variable(
            [CP_locs[0,:,:][None], CP_locs[-1,:,:][None]],
            trainable=False, name='ControlPointsUntrainable_ax0'
            )
        self.CP_locs_untrainable_ax1 = tf.Variable(
            [CP_locs[1:-1,0,:][:,None], CP_locs[1:-1,-1,:][:,None]],
            trainable=False, name='ControlPointsUntrainable_ax1'
            )
        
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.CP_idx_nn = (tf.Variable(CP_idx_nn, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
                          if CP_idx_nn is not None else {})
        
        self.A = tf.Variable([
            [-.5, 1.5, -1.5, 0.5],
            [1, -2.5, 2, -.5],
            [-.5, 0, .5, 0],
            [0, 1, 0, 0]
            ], trainable=False, dtype=tf.float32)
        
    
    @tf.function 
    def call(self, ch1, ch2):
        if self.direct:
            ch2_mapped = self.transform_vec(ch2)
            return tf.reduce_sum(tf.square(ch1-ch2_mapped)) 
        else:
            ch2_mapped = self.transform_mat(ch2)
            return Rel_entropy(ch1, ch2_mapped)
        
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        self.load_CPlocs()
        self.update_splines(self.CP_idx)        
        x = x_input[:,0][:,None]%1
        y = x_input[:,1][:,None]%1
        
        M_matrix = tf.stack([
            tf.pow(x,3)*tf.pow(y,3)*self.Sum_A(0,0),
            tf.pow(x,3)*tf.pow(y,2)*self.Sum_A(0,1),
            tf.pow(x,3)*tf.pow(y,1)*self.Sum_A(0,2),
            tf.pow(x,3)*tf.pow(y,0)*self.Sum_A(0,3),
            
            tf.pow(x,2)*tf.pow(y,3)*self.Sum_A(1,0),
            tf.pow(x,2)*tf.pow(y,2)*self.Sum_A(1,1),
            tf.pow(x,2)*tf.pow(y,1)*self.Sum_A(1,2),
            tf.pow(x,2)*tf.pow(y,0)*self.Sum_A(1,3),
        
            tf.pow(x,1)*tf.pow(y,3)*self.Sum_A(2,0),
            tf.pow(x,1)*tf.pow(y,2)*self.Sum_A(2,1),
            tf.pow(x,1)*tf.pow(y,1)*self.Sum_A(2,2),
            tf.pow(x,1)*tf.pow(y,0)*self.Sum_A(2,3),
            
            tf.pow(x,0)*tf.pow(y,3)*self.Sum_A(3,0),
            tf.pow(x,0)*tf.pow(y,2)*self.Sum_A(3,1),
            tf.pow(x,0)*tf.pow(y,1)*self.Sum_A(3,2),
            tf.pow(x,0)*tf.pow(y,0)*self.Sum_A(3,3),
            ], axis=2)
        return tf.reduce_sum(M_matrix, axis=2)
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_mat(self, x_input):
        self.load_CPlocs()
        self.update_splines(self.CP_idx_nn)        
        x = x_input[:,:,0][:,:,None]%1
        y = x_input[:,:,1][:,:,None]%1
        
        M_matrix = tf.stack([
            tf.pow(x,3)*tf.pow(y,3)*self.Sum_A(0,0),
            tf.pow(x,3)*tf.pow(y,2)*self.Sum_A(0,1),
            tf.pow(x,3)*tf.pow(y,1)*self.Sum_A(0,2),
            tf.pow(x,3)*tf.pow(y,0)*self.Sum_A(0,3),
            
            tf.pow(x,2)*tf.pow(y,3)*self.Sum_A(1,0),
            tf.pow(x,2)*tf.pow(y,2)*self.Sum_A(1,1),
            tf.pow(x,2)*tf.pow(y,1)*self.Sum_A(1,2),
            tf.pow(x,2)*tf.pow(y,0)*self.Sum_A(1,3),
        
            tf.pow(x,1)*tf.pow(y,3)*self.Sum_A(2,0),
            tf.pow(x,1)*tf.pow(y,2)*self.Sum_A(2,1),
            tf.pow(x,1)*tf.pow(y,1)*self.Sum_A(2,2),
            tf.pow(x,1)*tf.pow(y,0)*self.Sum_A(2,3),
            
            tf.pow(x,0)*tf.pow(y,3)*self.Sum_A(3,0),
            tf.pow(x,0)*tf.pow(y,2)*self.Sum_A(3,1),
            tf.pow(x,0)*tf.pow(y,1)*self.Sum_A(3,2),
            tf.pow(x,0)*tf.pow(y,0)*self.Sum_A(3,3),
            ], axis=3)
        return tf.reduce_sum(M_matrix, axis=3)
        
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def Sum_A(self,a,b):
        A_matrix = tf.stack([
            self.A[a,0]*self.A[b,0]*self.q00,
            self.A[a,0]*self.A[b,1]*self.q01,
            self.A[a,0]*self.A[b,2]*self.q02,
            self.A[a,0]*self.A[b,3]*self.q03,
            
            self.A[a,1]*self.A[b,0]*self.q10,
            self.A[a,1]*self.A[b,1]*self.q11,
            self.A[a,1]*self.A[b,2]*self.q12,
            self.A[a,1]*self.A[b,3]*self.q13,
            
            self.A[a,2]*self.A[b,0]*self.q20,
            self.A[a,2]*self.A[b,1]*self.q21,
            self.A[a,2]*self.A[b,2]*self.q22,
            self.A[a,2]*self.A[b,3]*self.q23,
            
            self.A[a,3]*self.A[b,0]*self.q30,
            self.A[a,3]*self.A[b,1]*self.q31,
            self.A[a,3]*self.A[b,2]*self.q32,
            self.A[a,3]*self.A[b,3]*self.q33
            ], axis=2)
        return tf.reduce_sum(A_matrix, axis=2)
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def load_CPlocs(self):
        self.CP_locs = tf.concat([ 
            self.CP_locs_untrainable_ax0[0],
            tf.concat([ 
                self.CP_locs_untrainable_ax1[0], 
                self.CP_locs_trainable,
                self.CP_locs_untrainable_ax1[1]
                ],axis=1),
            self.CP_locs_untrainable_ax0[1]
            ],axis=0)
        
    
    #@tf.function
    def update_splines(self, idx):
        self.q00 = tf.gather_nd(self.CP_locs, idx+[-1,-1])  # q_k
        self.q01 = tf.gather_nd(self.CP_locs, idx+[-1,0])  # q_k
        self.q02 = tf.gather_nd(self.CP_locs, idx+[-1,1])  # q_k
        self.q03 = tf.gather_nd(self.CP_locs, idx+[-1,2])  # q_k
            
        self.q10 = tf.gather_nd(self.CP_locs, idx+[0,-1])  # q_k
        self.q11 = tf.gather_nd(self.CP_locs, idx+[0,0])  # q_k
        self.q12 = tf.gather_nd(self.CP_locs, idx+[0,1])  # q_k
        self.q13 = tf.gather_nd(self.CP_locs, idx+[0,2])  # q_k
            
        self.q20 = tf.gather_nd(self.CP_locs, idx+[1,-1])  # q_k
        self.q21 = tf.gather_nd(self.CP_locs, idx+[1,0])  # q_k
        self.q22 = tf.gather_nd(self.CP_locs, idx+[1,1])  # q_k
        self.q23 = tf.gather_nd(self.CP_locs, idx+[1,2])  # q_k
            
        self.q30 = tf.gather_nd(self.CP_locs, idx+[2,-1])  # q_k
        self.q31 = tf.gather_nd(self.CP_locs, idx+[2,0])  # q_k
        self.q32 = tf.gather_nd(self.CP_locs, idx+[2,1])  # q_k
        self.q33 = tf.gather_nd(self.CP_locs, idx+[2,2])  # q_k
        
        
    def reset_CP(self, CP_idx, CP_idx_nn=None):
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.CP_idx_nn = (tf.Variable(CP_idx_nn, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
                          if CP_idx_nn is not None else {})