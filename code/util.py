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
    """
    Suppress TF warning message
    """
    import tensorflow as tf
    import os
    import logging
    from tensorflow.python.util import deprecation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    
class NormalizerClass(object):
    """
    Normalizer Class
    """
    def __init__(self,raw_data,eps=1e-8):
        self.raw_data = raw_data
        self.eps      = eps
        self.mu       = np.mean(self.raw_data,axis=0)
        self.std      = np.std(self.raw_data,axis=0)
        self.nzd_data = self.get_nzdval(self.raw_data)
        self.org_data = self.get_orgval(self.nzd_data)
        self.max_err  = np.max(self.raw_data-self.org_data)
    def get_nzdval(self,data):
        n = data.shape[0]
        nzd_data = (data - np.tile(self.mu,(n,1))) / np.tile(self.std+self.eps,(n,1))
        return nzd_data
    def get_orgval(self,data):
        n = data.shape[0]
        org_data = data*np.tile(self.std+self.eps,(n,1))+np.tile(self.mu,(n,1))
        return org_data
    
def get_mdn_training_data():
    """
    Get training data for mixture density networks
    """
    # Get x
    x_min,x_max,n_train_half,y_max,var_scale = 0,100,1000,100,1.0 # 0,100,1000,100,0.5 
    x_train = np.linspace(x_min,x_max,n_train_half).reshape((-1,1)) # [1000 x 1]
    
    # Shuffle?
    n = x_train.shape[0]
    x_train = x_train[np.random.permutation(n),:]

    # Get y
    y_train = np.concatenate((y_max*np.sin(2.0*np.pi*x_train/(x_max-x_min))+2*y_max*x_train/x_max,
                              y_max*np.cos(2.0*np.pi*x_train/(x_max-x_min)))+2*y_max*x_train/x_max,
                              axis=1) # [1000 x 2]
    x_train,y_train = np.concatenate((x_train,x_train),axis=0),np.concatenate((y_train,-y_train),axis=0)
    n_train = y_train.shape[0]
    y_train = y_train + var_scale*y_max*np.random.randn(n_train,2)*np.square(1-x_train/x_max) # add noise 
    nzr_x_train = NormalizerClass(x_train)
    x_train = nzr_x_train.get_nzdval(x_train) # normalize training input
    y_train = NormalizerClass(y_train).nzd_data # normalize training output 
    return x_train,y_train

    