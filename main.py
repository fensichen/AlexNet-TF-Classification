import numpy as np
import os
import sys
import math
import tensorflow as tf
import data 
import matplotlib.pyplot as plt

train_x       = np.zeros((1, 227, 227, 3), dtype = np.int32 )
train_y       = np.zeros((1, 1000))
x_dim         = train_x.shape[1:]
y_dim         = train_y.shape[1]

## Network parameter
num_input     = 1000 # img shape
dropout       = 0.5
std_dev       = math.sqrt(2/num_input)

## Training Parameters
LR            = 0.01
num_steps     = 20
batch_size    = 128
display_step  = 1

class ConvNet(object):
    ## constructor to build the model for the traininig ##
    def __init__(self, **kwargs ):

      # store layers weight & bias
      self.weights = {
          'wc1' : tf.Variable(tf.random_normal([11, 11, 3            , 96 ], stddev = 0.01)),
          'wc2' : tf.Variable(tf.random_normal([5,   5, 96           , 256], stddev = 0.01)),
          'wc3' : tf.Variable(tf.random_normal([3,   3, 256          , 384], stddev = 0.01)),
          'wc4' : tf.Variable(tf.random_normal([3,   3, 384          , 384], stddev = 0.01)),
          'wc5' : tf.Variable(tf.random_normal([3,   3, 384          , 256], stddev = 0.01)),
          'wfc1': tf.Variable(tf.random_normal([6*6*256              ,4096], stddev = 0.005)),
          'wfc2': tf.Variable(tf.random_normal([4096, 4096], stddev = 0.005)),
          'out' : tf.Variable(tf.random_normal([4096, 1000], stddev = 0.01))
      }

      self.biases = {
          'bc1' : tf.Variable(tf.zeros([96] )),
          'bc2' : tf.Variable(tf.ones([256] )),
          'bc3' : tf.Variable(tf.zeros([384] )),
          'bc4' : tf.Variable(tf.ones([384] )),
          'bc5' : tf.Variable(tf.ones([256] )),
          'bfc1': tf.Variable(tf.ones([4096] )),
          'bfc2': tf.Variable(tf.ones([4096] )),
          'out' : tf.Variable(tf.zeros([1000] ))
      }

      # Graph input
      self.X      = tf.placeholder(tf.float32, [None, 227, 227, 3] )
      self.p_drop = tf.placeholder( "float" )

    def conv2d( self, x, W, b, strides=1):
      # Conv2D wrapper, with bias and relu activation
      h = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') + b
      return tf.nn.relu(h)

    def maxpool2d( self, x, k, s):
      return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides = [1, s, s, 1], padding = 'VALID')
    
    def fc(self, x, W, b):
      h = tf.matmul(x, W) + b
      return tf.nn.relu(h)

    #Create the neural network
    def alex_net( self ):

      # Convolution Layer 1, stride = 4
      conv1    = self.conv2d( self.X, self.weights['wc1'], self.biases['bc1'], strides = 4 )
      # Max Pooling (down-sampling)
      pool1    = self.maxpool2d( conv1, k = 3, s = 2 )

      # Convolution Layer 2
      conv2    = self.conv2d( pool1, self.weights['wc2'], self.biases['bc2'], strides = 1 )
      pool2    = self.maxpool2d( conv2, k = 3, s = 2)

      # Convolution Layer 3
      conv3    = self.conv2d( pool2, self.weights['wc3'], self.biases['bc3'], strides = 1 )

      # Convolution Layer 4 
      conv4    = self.conv2d( conv3, self.weights['wc4'], self.biases['bc4'], strides = 1 )
     
      # Convolution Layer 5
      conv5    = self.conv2d( conv4, self.weights['wc5'], self.biases['bc5'], strides = 1)
      pool5    = self.maxpool2d( conv5, k = 3, s = 2)

      # FC5 : fully connectly layer 
      #pool_vec = tf.reshape(pool5, [pool5.get_shape().as_list()[0], -1]) 
     
      pool_vec = tf.reshape( pool5, [-1, 256*6*6] )
      #pool_vec = tf.convert_to_tensor([np.nan, 1, 1, 256*6*6])

      fc6      = tf.nn.dropout( self.fc(pool_vec, self.weights['wfc1'], self.biases['bfc1'] ), self.p_drop )
      # FC6
      fc7      = tf.nn.dropout( self.fc(fc6,   self.weights['wfc2'], self.biases['bfc2'] ), self.p_drop )
      # FC7 
      fc8      = tf.matmul( fc7, self.weights['out'] ) + self.biases['out']

      self.result = fc8

      return self.result


net          = ConvNet()
labels       = tf.placeholder( tf.int64, [None, 1000] )


# Construct model
logits       = net.alex_net()
prediction   = tf.nn.softmax( logits )

# Define loss and optimizer
loss_pre     = tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = labels )
loss         = tf.reduce_mean( loss_pre )

#optimizer    = tf.train.AdamOptimizer( learning_rate = LR )
# optimizer    = tf.train.GradientDescentOptimizer( LR )
# train_op     = optimizer.minimize(loss)
train_op     = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

# Evaluate model
#correct_pred = tf.equal( tf.argmax(prediction,1), tf.argmax(labels,1))
correct_pred = tf.equal( tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy     = tf.reduce_mean( tf.cast(correct_pred, tf.float32))

conf         = tf.ConfigProto( gpu_options = tf.GPUOptions( allow_growth = True ) )  

hnd_data_train = data.DataHandler()
hnd_data_train.parse('/home/fensi/nas/ImageNet/train.txt')
hnd_data_test  = data.DataHandler()
hnd_data_test.parse('/home/fensi/nas/ImageNet/test.txt')

# Start training
with tf.Session( config = conf ) as sess:

  # Initialize the variables 
  tf.global_variables_initializer().run()

  # Run the initializer
  num_steps = 1000
  
  for step in range(1, num_steps + 1):
    # Training
    hnd_data_train.shuffle()
    display = 0
    for start in range( 1, hnd_data_train.num() - hnd_data_train.num()% batch_size, batch_size ):
      end              = start + batch_size
      batch_x, batch_y = hnd_data_train.load_data( start, end, crop_size = 227, train = True ) 
      batch_x /= 128.
      s_batch_y        = np.zeros( (end-start, 1000) )
      for b in range( end-start ):
        s_batch_y[b,batch_y[b]] = 1

      # calculate batch loss and accuracy
      tf_loss, tf_loss_pre, acc, op, res = sess.run([loss, loss_pre, accuracy, train_op, net.result ], feed_dict={ net.X: batch_x, labels: s_batch_y, net.p_drop : 0.5 })  
      if display % display_step == 0:
        print ("Epoch " + str(step) + ", Iteration " + str(start).zfill(5) + "\tMinibatch Loss= " + "{:.4f}".format(tf_loss) + ", Training Accuracy = " + "{:.5f}".format(acc) )
      display += 1
        
    
    # Testing
    # for start, end in range( 1, hnd_data_test.num(), batch_size ):
    #     batch_x, batch_y = hnd_data_test.load_data( start, end, crop_size = 227, train = False ) 
    #     # calculate batch loss and accuracy
    #     loss, acc = sess.run([loss_op, accuracy], feed_dict={ net.X: batch_x, labels: batch_y, net.p_drop : 1 })
    #     print ("Step " + str(step) +", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy = " + "{:.35f}".format(acc) )
    
  print("Optimization Finished!")
        
