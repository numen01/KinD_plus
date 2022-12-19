import tensorflow as tf
import tensorflow.contrib.slim as slim

def illu_attention_3_M(input_feature, input_i, name):
  # Define kernel size and kernel initializer
  kernel_size = 3
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()

  # Define a variable scope with the given name
  with tf.variable_scope(name):
    # Apply a convolutional layer with kernel size 3 and no bias to input_i
    concat = tf.layers.conv2d(input_i,
                              filters=1,
                              kernel_size=[kernel_size,kernel_size],
                              strides=[1,1],
                              padding="same",
                              activation=None,
                              kernel_initializer=kernel_initializer,
                              use_bias=False,
                              name='conv')
    # Assert that the output of the convolutional layer has 1 feature channel
    assert concat.get_shape()[-1] == 1
    # Apply a sigmoid activation function to the output of the convolutional layer
    concat = tf.sigmoid(concat, 'sigmoid')

  # Return the element-wise product of input_feature and concat
  return input_feature * concat#, concat

def pool_upsamping_3_M(input_feature, level, training, name):
  # If level is 1
  if level == 1:
    # Define a variable scope with the given name
    with tf.variable_scope(name):
      # Apply a convolutional layer to input_feature
      pu_conv = slim.conv2d(input_feature, input_feature.get_shape()[-1], [3,3], 1, padding='SAME' ,scope='pu_conv')
      # Apply batch normalization to the output of the convolutional layer
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      # Apply a ReLU activation function to the output of batch normalization
      pu_conv = tf.nn.relu(pu_conv)
      # Assign the output of the ReLU activation to conv_up
      conv_up = pu_conv

  # If level is 2
  elif level == 2:
    # Define a variable scope with the given name
    with tf.variable_scope(name):
      # Apply max pooling with pooling size 2 and stride 2 to input_feature
      pu_net = slim.max_pool2d(input_feature, [2,2], 2, padding='SAME', scope='pu_net')
      # Apply a convolutional layer to the output of max pooling
      pu_conv = slim.conv2d(pu_net, input_feature.get_shape()[-1], [3,3], 1, padding='SAME' ,scope='pu_conv')
      # Apply batch normalization to the output of the convolutional layer
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      # Apply a ReLU activation function to the output of batch normalization
      pu_conv = tf.nn.relu(pu_conv)
      # Apply a transposed convolutional layer with stride 2 to the output of the ReLU activation
      conv_up = slim.conv2d_transpose(pu_conv, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up')

  # If level is 4
  elif level == 4:
    # Define a variable scope with the given name
    with tf.variable_scope(name):
      # Apply max pooling with pooling size 4 and stride 4 to input_feature
      pu_net = slim.max_pool2d(input_feature, [4,4], 4, padding='SAME', scope='pu_net')
      # Apply a convolutional layer to the output of max pooling
      pu_conv = slim.conv2d(pu_net, input_feature.get_shape()[-1], [1,1], 1, padding='SAME' ,scope='pu_conv')
      # Apply batch normalization to the output of the convolutional layer
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      # Apply a ReLU activation function to the output of batch normalization
      pu_conv = tf.nn.relu(pu_conv)
      # Apply a transposed convolutional layer with stride 2 to the output of the ReLU activation
      conv_up_1 = slim.conv2d_transpose(pu_conv, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up_1')
      # Apply a transposed convolutional layer with stride 2 to the output of the previous transposed convolutional layer
      conv_up = slim.conv2d_transpose(conv_up_1, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up')

  return conv_up

def Multi_Scale_Module_3_M(input_feature, training, name):
    # Apply pool_upsamping_3_M function with scale factor of 1 to input_feature
    Scale_1 = pool_upsamping_3_M(input_feature, 1, training, name=name+'pu1')
    # Apply pool_upsamping_3_M function with scale factor of 2 to input_feature
    Scale_2 = pool_upsamping_3_M(input_feature, 2, training, name=name+'pu2')
    # Apply pool_upsamping_3_M function with scale factor of 4 to input_feature
    Scale_4 = pool_upsamping_3_M(input_feature, 4, training, name=name+'pu4')

    # Concatenate input_feature, Scale_1, Scale_2, and Scale_4 along the channel axis
    res = tf.concat([input_feature, Scale_1, Scale_2, Scale_4], axis=3)
    # Apply a 1x1 convolution to res to produce the final multi-scale feature map
    multi_scale_feature = slim.conv2d(res, input_feature.shape[3], [1,1], 1, padding='SAME', scope=name+'multi_scale_feature')
    # Return the multi-scale feature map
    return multi_scale_feature

def msia_3_M(input_feature, input_i, name, training):
    # Apply illu_attention_3_M function to input_feature and input_i
    spatial_attention_feature = illu_attention_3_M(input_feature, input_i, name)
    # Apply Multi_Scale_Module_3_M function to spatial_attention_feature
    msia_feature = Multi_Scale_Module_3_M(spatial_attention_feature, training, name)
    # Return the msia_feature
    return msia_feature


