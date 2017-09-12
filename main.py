import numpy as np
import os
import sys
import math
import tensorflow as tf
import data 


train_x       = np.zeros((1, 227, 227, 3), dtype = np.int32 )
train_y       = np.zeros((1, 1000))
x_dim         = train_x.shape[1:]
y_dim         = train_y.shape[1]

## Network parameter
num_input     = 1000 # img shape
n_classes     = 4
dropout       = 0.5
std_dev       = math.sqrt(2/num_input)

## Training Parameters
learning_rate = 0.001
num_steps     = 200
batch_size    = 128
display_step  = 10 

class ConvNet(object):
    ## constructor to build the model for the traininig ##
    def __init__(self, **kwargs ):

        # initialize all allowed keys to false

        #if (self.dataset_training != False):
        #    self.train_imgs_lab = Dataset.load_dataset(self.dataset_training)
        #else: 
        #    self.test_imgs_lab = Dataset.load_dataset(self.dataset_test)

        # store layers weight & bias
        self.weights = {
            'wc1' : tf.Variable(tf.random_normal([11, 11, 3            , 96 ], stddev = std_dev)),
            'wc2' : tf.Variable(tf.random_normal([5,   5, 96           , 256], stddev = std_dev)),
            'wc3' : tf.Variable(tf.random_normal([3,   3, 256          , 384], stddev = std_dev)),
            'wc4' : tf.Variable(tf.random_normal([3,   3, 384          , 384], stddev = std_dev)),
            'wc5' : tf.Variable(tf.random_normal([3,   3, 384          , 256], stddev = std_dev)),
            'wfc1': tf.Variable(tf.random_normal([6*6*256              ,4096], stddev = std_dev)),
            'wfc2': tf.Variable(tf.random_normal([4096, 4096], stddev = std_dev)),
            'out' : tf.Variable(tf.random_normal([4096, 1000], stddev = std_dev))
        }

        self.biases = {
            'bc1' : tf.Variable(tf.random_normal([96],   stddev = 0 )),
            'bc2' : tf.Variable(tf.random_normal([256],  stddev = 0 )),
            'bc3' : tf.Variable(tf.random_normal([384],  stddev = 0 )),
            'bc4' : tf.Variable(tf.random_normal([384],  stddev = 0 )),
            'bc5' : tf.Variable(tf.random_normal([256],  stddev = 0 )),
            'bfc1': tf.Variable(tf.random_normal([4096], stddev = 0 )),
            'bfc2': tf.Variable(tf.random_normal([4096], stddev = 0 )),
            'out' : tf.Variable(tf.random_normal([1000], stddev = 0 ))
        }

        # Graph input
        self.X      = tf.placeholder(tf.float32, [None, 227, 227, 3] )

    def conv2d( self, x, W, b, strides=1):
      # Conv2D wrapper, with bias and relu activation
      x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
      x = tf.nn.bias_add(x, b)
      return tf.nn.relu(x)

    def maxpool2d( self, x, k, s):
      return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides = [1, s, s, 1], padding = 'VALID')
    
    def fc(self, x, W, b):
      x = tf.matmul(x, W) + b
      return tf.nn.relu(x)

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
      FC1      = self.fc(pool_vec, self.weights['wfc1'], self.biases['bfc1'] )
      # FC6
      FC2      = self.fc(FC1,   self.weights['wfc2'], self.biases['bfc2'] )
      # FC7 
      output   = self.fc(FC2,   self.weights['out'],   self.biases['out'] )

      return output


net          = ConvNet()
labels       = tf.placeholder( tf.int64, [None] )


# Construct model
logits       = net.alex_net()
prediction   = tf.nn.softmax( logits )

# Define loss and optimizer
loss_op      = tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logits, labels = labels )
loss_op      = tf.reduce_mean( loss_op )

optimizer    = tf.train.AdamOptimizer( learning_rate = learning_rate )
train_op     = optimizer.minimize(loss_op)

# Evaluate model
#correct_pred = tf.equal( tf.argmax(prediction,1), tf.argmax(labels,1))
correct_pred = tf.equal( tf.argmax(prediction,1), labels)
accuracy     = tf.reduce_mean( tf.cast(correct_pred, tf.float32))

conf   = tf.ConfigProto( 
                                                    gpu_options = tf.GPUOptions( allow_growth = True ) 
                        )  

hnd_data_train = data.DataHandler()
hnd_data_train.parse('/home/fensi/nas/ImageNet/train.txt')
hnd_data_test  = data.DataHandler()
hnd_data_test.parse('/home/fensi/nas/ImageNet/test.txt')

# Start training
with tf.Session( config = conf ) as sess:

  # Initialize the variables 
  init         = tf.global_variables_initializer()

  # Run the initializer
  sess.run(init)
  num_steps = 1000
  for step in range(1, num_steps + 1):
    # Training
    #for start, end in range( 1, hnd_data_train.num(), batch_size ):
    for start, end in zip( range( 1, 1281152, batch_size ), range( batch_size, 1281152, batch_size )) :
        batch_x, batch_y = hnd_data_train.load_data( start, end, crop_size = 227, train = True ) 
        # calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={ net.X: batch_x, labels: batch_y })
        print ("Step " + str(step) +", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc) )
    
    
    # Testing
    for start, end in range( 1, hnd_data_test.num(), batch_size ):
        batch_x, batch_y = hnd_data_test.load_data( start, end, crop_size = 227, train = False ) 
        # calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={ net.X: batch_x, labels: batch_y })
        print ("Step " + str(step) +", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy = " + "{:.3f}".format(acc) )
    
  print("Optimization Finished!")
        
