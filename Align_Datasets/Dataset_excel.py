# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import copy
import pandas as pd

from AlignModel import AlignModel
from Align_Datasets.channel_class import channel


class Dataset_excel(AlignModel):
    def __init__(self, path, pix_size=1, align_rcc=True, coupled=False, 
                 imgshape=[512, 512], shift_rcc=None):
        self.pix_size=pix_size
        self.imgshape=imgshape
        self.shift_rcc=shift_rcc
        self.align_rcc=align_rcc
        self.coupled=coupled
        self.ch1, self.ch2 = self.load_dataset(path)
        self.ch2_original=copy.deepcopy(self.ch2)
        self.img, self.imgsize, self.mid = self.imgparams()                     # loading the image parameters
        self.center_image()
        
        AlignModel.__init__(self)
        
        
    #%% functions
    def load_dataset(self,path, shift_rcc=None):
        data = pd.read_csv(path)
        grouped = data.groupby(data.Channel)
        ch1 = grouped.get_group(1)
        ch2 = grouped.get_group(2)
        
        data1 = np.array(ch1[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data1 = np.column_stack((data1, np.arange(data1.shape[0])))
        data2 = np.array(ch2[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
        data2 = np.column_stack((data2, np.arange(data2.shape[0])))
    
        ch1 = channel(self.imgshape, pos = np.float32(data1[:,:2]), frame = data1[:,2], index = data1[:,4])
        ch2 = channel(self.imgshape, pos = np.float32(data2[:,:2]), frame = data2[:,2], index = data2[:,4])
        self.Nbatch = 1
        if self.align_rcc is not False:
            if shift_rcc is None:
                shift_rcc=ch1.align(ch2)
                print('Shifted with RCC of', shift_rcc)  
            ch1.pos += shift_rcc 
        ch1.pos *= self.pix_size
        ch2.pos *= self.pix_size
           
        return [ch1], [ch2]
    
    
    def imgparams(self):
    # calculate borders of system
    # returns a 2x2 matrix containing the edges of the image, a 2-vector containing
    # the size of the image and a 2-vector containing the middle of the image
        img1 = np.empty([2,2, self.Nbatch], dtype = float)
        for batch in range(self.Nbatch):
            img1[0,0,batch] = np.min(( np.min(self.ch1[batch].pos[:,0]), np.min(self.ch2[batch].pos[:,0]) ))
            img1[1,0,batch] = np.max(( np.max(self.ch1[batch].pos[:,0]), np.max(self.ch2[batch].pos[:,0]) ))
            img1[0,1,batch] = np.min(( np.min(self.ch1[batch].pos[:,1]), np.min(self.ch2[batch].pos[:,1]) ))
            img1[1,1,batch] = np.max(( np.max(self.ch1[batch].pos[:,1]), np.max(self.ch2[batch].pos[:,1]) ))
        
        img = np.empty([2,2], dtype = float)
        img[0,0] = np.min(img1[0,0,:])
        img[1,0] = np.max(img1[1,0,:])
        img[0,1] = np.min(img1[0,1,:])
        img[1,1] = np.max(img1[1,1,:])
        return img, (img[1,:] - img[0,:]), (img[1,:] + img[0,:])/2
    
    
    def center_image(self):
        if self.mid is None: self.img, self.imgsize, self.mid = self.imgparams() 
        for batch in range(self.Nbatch):
            self.ch1[batch].pos -= self.mid
            self.ch2[batch].pos -= self.mid
            self.ch2_original[batch].pos -= self.mid
        self.img, self.imgsize, self.mid = self.imgparams() 
