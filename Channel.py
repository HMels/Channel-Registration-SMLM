# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:47:49 2021

@author: Mels
"""
import tensorflow as tf

#%% Channel 
class Channel:
    def __init__(self, pos=None, frame=None):
        self.pos = [tf.Variable(pos, dtype=tf.float32, trainable=False)] if pos is not None else {}
        self.frame = [tf.Variable(frame, dtype=tf.float32, trainable=False)] if frame is not None else {}
        self.img, self.imgsize, self.mid = self.imgparams()
        
        
    def pos_all(self):
        return tf.concat(self.pos, axis=0)
        
    
    def imgparams(self):
        pos = self.pos_all()
        img = tf.Variable([[ tf.reduce_min(pos[:,0]), tf.reduce_min(pos[:,1]) ],
                           [ tf.reduce_max(pos[:,0]),  tf.reduce_max(pos[:,1]) ]], dtype=tf.float32)
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
    
    
    def SplitFrames(self):
        print('Splitting Dataset into different frames...')
        if len(self.pos)>1: raise ValueError('Dataset already split in batches')
        frames,_ = tf.unique(self.frame[0])
        (pos,frame)=([],[])
        for fr in frames:
            idx = tf.where(self.frame[0]==fr)
            pos.append( tf.squeeze(tf.gather(self.pos[0],idx,axis=0), axis=1) )
            frame.append( self.frame[0][idx] )
        self.pos = pos
        self.frame = frame
        
        
    def load_NN_matrix(self, idxlist):
    # Takes the indexes for channel 1 and 2 and loads the matrix
        if len(self.pos)!=len(idxlist): raise ValueError('idxlist should be same size as the positions')
        self.NNpos=[]
        for batch in range(len(self.pos)):
            NN=[]
            for nn in idxlist[batch]:
                NN.append(tf.gather(self.pos[batch],nn,axis=0))
            self.NNpos.append( tf.stack(NN, axis=0) )