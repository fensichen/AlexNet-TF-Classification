import sys
import numpy as np
import os
import tensorflow as tf
import cv2 as cv
import random

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
  def shuffle( self, filenames ):

    # generate a random permutation
    self.perm     = range( len( self.list ) )
    random.shuffle( self.perm )
  
  def num(self):
    return len(self.list)

  def load_data( self, start, end, crop_size = 224, train = False ):
    #fensi
    self.perm     = range( len( self.list ) )
    perm          = self.perm
    list          = self.list
    
    # Probe file to determin dimension
    print(list[  perm[start]  ])
    probe    = cv.imread(list[ perm[start] ])

    # Convert shape
    height, width, channels = probe.shape

    # Create data array
    data  = np.zeros( (end-start, crop_size, crop_size, probe.shape[-1]), dtype = np.float32)
    label = np.zeros( (end-start) )

    for i in range( start, end ):
      img = cv.imread ( list[ perm[i] ])
      img = np.reshape( img, [img.shape[0], img.shape[1], -1] )
  
      if train == False:
        hflip = 0
      else:
        hflip = random.randint(0, 1)

      # resize to 256
      factor =  256. / min( img.shape[0], img.shape[1] )
      img    = cv.resize( img, None, fx=factor, fy=factor )

      shape  = img.shape
      
      if train == False: # one center crop for testing 
        cc_x = (shape[1] - crop_size)/2   
        cc_y = (shape[0] - crop_size)/2
      else:
        cc_x = random.randint( 0, shape[1] - crop_size ) # crop bottom-left corner 
        cc_y = random.randint( 0, shape[0] - crop_size ) # crop 
    
      img = img[ cc_y : cc_y+crop_size, cc_x : cc_x + crop_size, :] 
            
      if hflip == 1:
        img = img[:,::-1,:]

      data[i-start]  = img
      label[i-start] = int( self.labels[ perm[i] ] )
        
    return data, label
    
    
    
    