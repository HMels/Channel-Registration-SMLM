# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:47:49 2021

@author: Mels
"""
import tensorflow as tf

#%% Channel 
class Channel:
    def __init__(self, pos=None, frame=None, group=None):
        self.pos = tf.Variable(pos, dtype=tf.float32, trainable=False) if pos is not None else {}
        self.frame = (tf.Variable(frame, dtype=tf.float32, trainable=False) if frame is not None else 
                      tf.ones(pos.shape[0], dtype=tf.float32))
        self.group = tf.Variable(group, dtype=tf.int32, trainable=False) if group is not None else tf.zeros(self.frame.shape[0], dtype=tf.int32) 
        
        if self.pos.shape[0]!=self.frame.shape[0]: raise ValueError('Frame and Positions are not equal in size!')
        self.img, self.imgsize, self.mid = self.imgparams()
        
        
    def pos_all(self):
        return self.pos.numpy() #tf.concat(self.pos, axis=0)
        
    
    def imgparams(self):
        pos = self.pos_all()
        if len(pos.shape)==3: pos=pos[:,0,:]
        elif len(pos.shape)!=2: raise ValueError('Invalid input shape! ch1 has shape '+str(pos.shape) )
        
        img = tf.Variable([[ tf.reduce_min(pos[:,0]), tf.reduce_min(pos[:,1]) ],
                           [ tf.reduce_max(pos[:,0]),  tf.reduce_max(pos[:,1]) ]], dtype=tf.float32)
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
        
      
    def center(self):
        self.pos.assign(self.pos - tf.reduce_mean(self.pos,axis=0))
        
    def offset(self, os):
        if not isinstance(os, list): raise ValueError('Invalid input. Offset must be a list!')
        self.pos=tf.stack([self.pos[:,0]+os[0],self.pos[:,1]+os[1]],axis=1)
        
        
    def AppendChannel(self, other):
        pos=self.pos
        frame=self.frame
        group=self.group
        del self.pos,self.frame, self.group
        self.pos=tf.Variable(tf.concat([pos,other.pos],axis=0), dtype=tf.float32, trainable=False) 
        self.frame=tf.Variable(tf.concat([frame,other.frame],axis=0), dtype=tf.float32, trainable=False) 
        self.group=tf.Variable(tf.concat([group,other.group],axis=0), dtype=tf.int32, trainable=False) 
        
        
    def ClusterCOM(self):
        clust=[]
        clustlist=tf.unique(self.group)[0]
        for i in clustlist:
            pos=tf.gather_nd(self.pos, tf.where(self.group==i))
            clust.append(tf.reduce_sum(pos,axis=0)/pos.shape[0])
        if len(clust)!=len(clustlist): raise Exception('The clusters have not been sorted!')
        return tf.stack(clust, axis=0), clustlist
        

    def transpose_axis(self):
        self.pos.assign(tf.Variable(self.pos.numpy()[:,[1,0]], dtype=tf.float32, trainable=False))
        
    def mirror_xaxis(self):
        self.pos.assign(tf.stack([-self.pos.numpy()[:,0],self.pos.numpy()[:,1]], axis=1))
        
    def mirror_yaxis(self):
        self.pos.assign(tf.stack([self.pos.numpy()[:,0],-self.pos.numpy()[:,1]], axis=1))