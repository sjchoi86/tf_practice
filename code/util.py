import numpy as np
import tensorflow as tf

def mlp(x,h_dims=[256,256],actv=tf.nn.relu,out_actv=tf.nn.relu,
        USE_DROPOUT=False,ph_is_training=None):
    """
    Multi-layer perceptron 
    """
    ki = tf.truncated_normal_initializer(stddev=0.1)
    for h_dim in h_dims[:-1]:
        x = tf.layers.dense(x,units=h_dim,activation=actv,kernel_initializer=ki)
        if USE_DROPOUT:
            x = tf.layers.dropout(x,rate=0.5,training=ph_is_training)
    return tf.layers.dense(x,units=h_dims[-1],activation=out_actv,kernel_initializer=ki)

def placeholder(dim=None):
    """
    Placeholder
    """
    return tf.placeholder(dtype=tf.float32,shape=(None,dim) if dim else (None,))

def placeholders(*args):
    """
    Usage: a_ph,b_ph,c_ph = placeholders(adim,bdim,None)
    """
    return [placeholder(dim) for dim in args]

def get_vars(scope):
    """
    Get TF variables within scope
    """
    if tf.__version__ == '1.12.0':
        tf_vars = [x for x in tf.global_variables() if scope in x.name]
    else:
        tf_vars = [x for x in tf.compat.v1.global_variables() if scope in x.name]
    return tf_vars

def get_mnist():
    """
    Get MNIST
    """
    mnist = tf.keras.datasets.mnist 
    (x_train,y_train),(x_test,y_test) = mnist.load_data() # 0~255
    y_train,y_test = np.eye(10)[y_train],np.eye(10)[y_test]
    x_train,x_test = x_train.reshape((-1,784)),x_test.reshape((-1,784)) # reshape [N x 784]
    x_train,x_test = x_train/255.0,x_test/255.0 # pixel values between 0 and 1
    return x_train,y_train,x_test,y_test

def gpu_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

def suppress_tf_warning():
    import tensorflow as tf
    import os
    import logging
    from tensorflow.python.util import deprecation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    deprecation._PRINT_DEPRECATION_WARNINGS = False