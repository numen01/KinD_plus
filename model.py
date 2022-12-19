import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from msia_BN_3_M import *

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    # Create a new variable scope
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        # Set the pooling size and define the deconvolution filter
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        # Perform the deconvolution operation
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)
        # Concatenate the deconvolved tensor with x2 along the channel dimension
        deconv_output =  tf.concat([deconv, x2],3)
        # Set the shape of the resulting tensor
        deconv_output.set_shape([None, None, None, output_channels*2])
        # Return the resulting tensor
        return deconv_output

def DecomNet(input):
    # Create a variable scope for the DecomNet network
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        # Apply a convolutional layer to the input image with 32 filters of size 3x3 and a stride of 1
        conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        # Apply a max pooling layer with a pooling size of 2x2 and a stride of 2
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        # Apply another convolutional layer with 64 filters of size 3x3 and a stride of 1
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        # Apply another max pooling layer with a pooling size of 2x2 and a stride of 2
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        # Apply a third convolutional layer with 128 filters of size 3x3 and a stride of 1
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        # Perform an upsampling and concatenation operation using conv3 and conv2
        up8 =  upsample_and_concat( conv3, conv2, 64, 128 , 'g_up_1')
        # Apply a convolutional layer with 64 filters of size 3x3 and a stride of 1
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        # Perform another upsampling and concatenation operation using conv8 and conv1
        up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'g_up_2')
        # Apply a convolutional layer with 32 filters of size 3x3 and a stride of 1
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        # Apply a convolutional layer with 3 filters of size 1x1 and a stride of 1
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        conv10=slim.conv2d(conv9,3,[1,1], rate=1, activation_fn=None, scope='g_conv10')
        # Apply a sigmoid activation function to obtain the reflectance image
        R_out = tf.sigmoid(conv10)

        # Apply a convolutional layer with 32 filters of size 3x3 and a stride of 1 to the output of the first max pooling layer
        l_conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='l_conv1_2')
        # Concatenate l_conv2 with conv9 along the channel dimension
        l_conv3=tf.concat([l_conv2, conv9],3)
        # Apply a convolutional layer with 1 filter of size 1x1 and a stride of 1
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        l_conv4=slim.conv2d(l_conv3,1,[1,1], rate=1, activation_fn=None,scope='l_conv1_4')
        # Apply a sigmoid activation function to obtain the lighting image
        L_out = tf.sigmoid(l_conv4)

    # Return the reflectance image and the lighting image
    return R_out, L_out


def Restoration_net(input_r, input_i, training = True):
    # Create a variable scope for the Denoise_Net network
    with tf.variable_scope('Denoise_Net', reuse=tf.AUTO_REUSE):
        # Apply a convolutional layer to input_r with 32 filters of size 3x3 and a stride of 1
        conv1=slim.conv2d(input_r, 32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_1')
        # Apply another convolutional layer to conv1 with 64 filters of size 3x3 and a stride of 1
        conv1=slim.conv2d(conv1,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_2')
        # Apply the msia_3_M function to conv1 and input_i
        msia_1 = msia_3_M(conv1, input_i, name='de_conv1', training=training)#, name='de_conv1_22')

        # Apply a convolutional layer to msia_1 with 128 filters of size 3x3 and a stride of 1
        conv2=slim.conv2d(msia_1,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv2_1')
        # Apply another convolutional layer to conv2 with 256 filters of size 3x3 and a stride of 1
        conv2 = slim.conv2d(conv2, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_2')
        # Apply the msia_3_M function to conv2 and input_i
        msia_2 = msia_3_M(conv2, input_i, name='de_conv2', training=training)

        # Apply a convolutional layer to msia_2 with 512 filters of size 3x3 and a stride of 1
        conv3=slim.conv2d(msia_2,512,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_1')
        # Apply another convolutional layer to conv3 with 256 filters of size 3x3 and a stride of 1
        conv3=slim.conv2d(conv3,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_2')
        # Apply the msia_3_M function to conv3 and input_i
        msia_3 = msia_3_M(conv3, input_i, name='de_conv3', training=training)

        conv4=slim.conv2d(msia_3,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_1')
        conv4=slim.conv2d(conv4,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_2')
        msia_4 = msia_3_M(conv4, input_i, name='de_conv4', training=training)

        conv5=slim.conv2d(msia_4,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv5_1')
        conv10=slim.conv2d(conv5,3,[3,3], rate=1, activation_fn=None, scope='de_conv10')
        out = tf.sigmoid(conv10)
        return out


def Illumination_adjust_net(input_i, input_ratio):
    # Create a variable scope for the I_enhance_Net network
    with tf.variable_scope('I_enhance_Net', reuse=tf.AUTO_REUSE):
        # Concatenate input_i and input_ratio along the channel dimension
        input_all = tf.concat([input_i, input_ratio], 3)

        # Apply a convolutional layer to input_all with 32 filters of size 3x3 and a stride of 1
        conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv_1')
        # Apply another convolutional layer to conv1 with 32 filters of size 3x3 and a stride of 1
        conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv_2')
        # Apply another convolutional layer to conv2 with 32 filters of size 3x3 and a stride of 1
        conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv_3')
        # Apply another convolutional layer to conv3 with 1 filter of size 3x3 and a stride of 1
        conv4 = slim.conv2d(conv3, 1, [3, 3], rate=1, activation_fn=lrelu, scope='conv_4')

        # Apply a sigmoid activation function to the output of conv4 to obtain the enhanced lighting image
        L_enhance = tf.sigmoid(conv4)
    # Return the enhanced lighting image
    return L_enhance
