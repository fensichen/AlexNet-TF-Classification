import sys
import numpy as np
import os
import tensorflow as tf
import random
import scipy as scp
import scipy.misc
import math

# FlipH can be done with python code:
# data.shape == H x W x C
# data = data[:,::-1,:]


class DataHandler():
  def __init__( self ):
    self.list     = []
    self.labels   = []

  def parse( self, filenames ):
    images        = [line.strip().split(' ')[0] for line in open( filenames )]
    file_length   = len( images )
    labels        = [line.strip().split(' ')[1] for line in open( filenames )] 
    
    # concatenate the list `file` with the overall `list` 
    self.list     = self.list + images[:]  
    self.labels   = self.labels + labels[:]

  # filenames 
  def shuffle( self ):
    # generate a random permutation
    self.perm     = range( len( self.list ) )
    random.shuffle( self.perm )
  
  def num(self):
    return len(self.list)

  def load_data( self, start, end, crop_size = 224, train = False ):
    
    perm          = self.perm
    list          = self.list

    # Probe file to determin dimension
    probe         = scipy.misc.imread( list[ self.perm[start] ] )

    # Create data array
    data          = np.zeros( (end-start, crop_size, crop_size, 3), dtype = np.float32)
    label         = np.zeros( (end-start), dtype = np.int64 )
    mean          = np.array( [104.,117.,123.] ) # mean for ImageNet
    
    for i in range( start, end ):
    
      img         = scipy.misc.imread ( list[ perm[i] ])
      img         = np.reshape( img, [img.shape[0], img.shape[1], -1] )  
      img         = img.astype( np.float32 ) # this step is important, because we need to subtract the mean later

      if img.shape[2] == 1: # if it is gray image, make it a color image by stacking channels
        img_tmp          = np.zeros( ( img.shape[0], img.shape[1], 3))
        img_tmp[:,:,0:1] = img
        img_tmp[:,:,1:2] = img
        img_tmp[:,:,2:3] = img
        img              = img_tmp 

      if img.shape[2] == 4:
        img              = img[:,:,0:3]

      # start data augmentation: flipping
      if train == False:
        hflip            = 0
      else:
        hflip            = random.randint(0, 1)

      # resize the shortest dimention to 256 pixel
      factor      =  256. / min( img.shape[0], img.shape[1] )
      new_shape   = [ int (img.shape[0] * factor), int (img.shape[1] * factor), img.shape[2] ]
      img         = scipy.misc.imresize( img, new_shape )
      shape       = img.shape
      
      # find the cropping center 
      if train == False: # one center crop for testing 
        cc_x            = (shape[1] - crop_size)/2   
        cc_y            = (shape[0] - crop_size)/2
      else:
        cc_x            = random.randint( 0, shape[1] - crop_size ) # crop bottom-left corner 
        cc_y            = random.randint( 0, shape[0] - crop_size ) # crop 
    
      img               = img[ cc_y : cc_y+crop_size, cc_x : cc_x + crop_size, :] 
      
      if hflip == 1:
        img             = img[:,::-1,:]


      #substract the mean
      img               = img - mean 
      data[i-start]     = img
      label[i-start]    = int( self.labels[ perm[i] ] )
        
    return data, label
    
    
    
    