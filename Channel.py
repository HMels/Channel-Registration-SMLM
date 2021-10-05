# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:47:49 2021

@author: Mels
"""
import tensorflow as tf

#%% Channel 
class Channel:
    def __init__(self, pos=None, frame=None):
        self.pos = tf.Variable(pos, dtype=tf.float32, trainable=False) if pos is not None else {}
        self.frame = tf.Variable(frame, dtype=tf.float32, trainable=False) if frame is not None else {}
        self.img, self.imgsize, self.mid = self.imgparams()
        
        
    def pos_all(self):
        return self.pos.numpy() #tf.concat(self.pos, axis=0)
        
    
    def imgparams(self):
        pos = self.pos_all()
        img = tf.Variable([[ tf.reduce_min(pos[:,0]), tf.reduce_min(pos[:,1]) ],
                           [ tf.reduce_max(pos[:,0]),  tf.reduce_max(pos[:,1]) ]], dtype=tf.float32)
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
        
       