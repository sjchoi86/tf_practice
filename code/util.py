import math,ray,os,time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf 
from scipy.spatial import distance
import scipy.optimize
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
    
def suppress_gym_warning():
    """
    Suppress gym warning message
    """
    import gym
    gym.logger.set_level(40) # gym logger 
    
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

def cos_exp_square_nd(x):
    """
    f(x) = -cos(2*pi*||x||_2)*exp(-||x||_2)
        x: [N x d]
    """
    x_rownorm = np.linalg.norm(x,axis=1).reshape((-1,1)) # [N x 1]
    y = -np.cos(2*np.pi*x_rownorm)*np.exp(-x_rownorm**1) # [N x 1]
    return y

def plot_line(
    x,y,fmt='-',lc='k',lw=2,label=None,
    x2=None,y2=None,fmt2='-',lc2='k',lw2=2,ms2=12,mfc2='none',mew2=2,label2=None,
    x3=None,y3=None,fmt3='-',lc3='k',lw3=2,ms3=12,mfc3='none',mew3=2,label3=None,
    x4=None,y4=None,fmt4='-',lc4='k',lw4=2,ms4=12,mfc4='none',mew4=2,label4=None,
    x5=None,y5=None,fmt5='-',lc5='k',lw5=2,ms5=12,mfc5='none',mew5=2,label5=None,
    x6=None,y6=None,fmt6='-',lc6='k',lw6=2,ms6=12,mfc6='none',mew6=2,label6=None,
    x7=None,y7=None,fmt7='-',lc7='k',lw7=2,ms7=12,mfc7='none',mew7=2,label7=None,
    x_fb=None,y_fb_low=None,y_fb_high=None,fba=0.1,fbc='g',labelfb=None,
    figsize=(10,5),
    xstr='',xfs=12,ystr='',yfs=12,
    tstr='',tfs=15,
    ylim=None,
    lfs=15,lloc='lower right'):
    """
    Plot a line
    """
    plt.figure(figsize=figsize)
    plt.plot(x,y,fmt,color=lc,linewidth=lw,label=label)
    if (x2 is not None):
        plt.plot(x2,y2,fmt2,color=lc2,linewidth=lw2,ms=ms2,mfc=mfc2,mew=mew2,label=label2)
    if (x3 is not None):
        plt.plot(x3,y3,fmt3,color=lc3,linewidth=lw3,ms=ms3,mfc=mfc3,mew=mew3,label=label3)
    if (x4 is not None):
        plt.plot(x4,y4,fmt4,color=lc4,linewidth=lw4,ms=ms4,mfc=mfc4,mew=mew4,label=label4)
    if (x5 is not None):
        plt.plot(x5,y5,fmt5,color=lc5,linewidth=lw5,ms=ms5,mfc=mfc5,mew=mew5,label=label5)
    if (x6 is not None):
        plt.plot(x6,y6,fmt6,color=lc6,linewidth=lw6,ms=ms6,mfc=mfc6,mew=mew6,label=label6)
    if (x7 is not None):
        plt.plot(x7,y7,fmt7,color=lc7,linewidth=lw7,ms=ms7,mfc=mfc7,mew=mew7,label=label7)

    if (x_fb is not None):
        plt.fill_between(x_fb.reshape(-1),
                        (y_fb_low).reshape(-1),
                        (y_fb_high).reshape(-1),
                        alpha=fba,color=fbc,label=labelfb)

    plt.xlabel(xstr,fontsize=xfs)
    plt.ylabel(ystr,fontsize=yfs)
    plt.title(tstr,fontsize=tfs)

    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])

    plt.legend(fontsize=lfs,loc=lloc)
    plt.show()
    
def x_sampler(n_sample,x_minmax):
    """
    Sample x as a list from the input domain 
    """
    x_samples = []
    for _ in range(n_sample):
        x_sample = x_minmax[:,0]+(x_minmax[:,1]-x_minmax[:,0])*np.random.rand(1,x_minmax.shape[0])
        x_samples.append(x_sample)
    return x_samples # list 

def sqrt_safe(x,eps=1e-6):
    return np.sqrt(np.abs(x)+eps)

def r_sq(x1,x2,x_range=1.0,invlen=5.0):
    """
    Scaled pairwise dists 
    """
    x1_scaled,x2_scaled = invlen*x1/x_range,invlen*x2/x_range
    D_sq = distance.cdist(x1_scaled,x2_scaled,'sqeuclidean') 
    return D_sq
    
def k_m52(x1,x2,x_range=1.0,gain=1.0,invlen=5.0):
    """
    Automatic relevance determination (ARD) Matern 5/2 kernel
    """
    R_sq = r_sq(x1,x2,x_range=x_range,invlen=invlen)
    eps = 1e-6
    K = gain*(1+sqrt_safe(5*R_sq)+(5.0/3.0)*R_sq)*np.exp(-sqrt_safe(5*R_sq))
    return K

def gp_m52(x,y,x_test,x_minmax=None,gain=1.0,invlen=5.0,eps=1e-8):
    """
    Gaussian process with ARD Matern 5/2 Kernel
    """
    if x_minmax is None:
        x_range = np.max(x,axis=0)-np.min(x,axis=0)
    else:
        xmin,xmax = x_minmax[:,0],x_minmax[:,1]
        x_range = xmax-xmin
    
    k_test = k_m52(x_test,x,x_range=x_range,gain=gain,invlen=invlen)
    K = k_m52(x,x,x_range=x_range,gain=gain,invlen=invlen)
    n = x.shape[0]
    inv_K = np.linalg.inv(K+eps*np.eye(n))
    mu_y = np.mean(y)
    mu_test = np.matmul(np.matmul(k_test,inv_K),y-mu_y)+mu_y
    var_test = (gain-np.diag(np.matmul(np.matmul(k_test,inv_K),k_test.T))).reshape((-1,1))
    return mu_test,var_test

def Phi(x):
    """
    CDF of Gaussian
    """
    return (1.0 + erf(x / math.sqrt(2.0))) / 2.0
    
def acquisition_function(x_bo,y_bo,x_test,x_minmax=None,SCALE_Y=True,gain=1.0,invlen=5.0,eps=1e-6):
    """
    Acquisition function of Bayesian Optimization with Expected Improvement
    """
    if SCALE_Y:
        y_bo_scaled = np.copy(y_bo)
        y_bo_mean = np.mean(y_bo_scaled)
        y_bo_scaled = y_bo_scaled - y_bo_mean
        y_min,y_max = np.min(y_bo_scaled), np.max(y_bo_scaled)
        y_range = y_max - y_min
        y_bo_scaled = 2.0 * y_bo_scaled / y_range
    else:
        y_bo_scaled = np.copy(y_bo)
    
    mu_test,var_test = gp_m52(x_bo,y_bo_scaled,x_test,x_minmax=x_minmax,gain=gain,invlen=invlen,eps=eps)
    gamma = (np.min(y_bo_scaled) - mu_test)/sqrt_safe(var_test)
    a_ei = 2.0 * sqrt_safe(var_test) * (gamma*Phi(gamma) + norm.pdf(mu_test,0,1))
    
    if SCALE_Y:
        mu_test = 0.5 * y_range * mu_test + y_bo_mean
    
    return a_ei,mu_test,var_test

def scale_to_match_range(x_to_change,y_to_refer):
    """
    Scale the values of 'x_to_change' to match the range of 'y_to_refer'
    """
    x_to_change_scale = np.copy(x_to_change)
    xmin,xmax = np.min(x_to_change_scale),np.max(x_to_change_scale)
    ymin,ymax = np.min(y_to_refer),np.max(y_to_refer)
    x_to_change_scale = (ymax-ymin)*(x_to_change_scale-xmin)/(xmax-xmin)+ymin
    return x_to_change_scale

def get_best_xy(x_data,y_data):
    """
    Get the current best solution
    """
    min_idx = np.argmin(y_data)
    return x_data[min_idx,:].reshape((1,-1)),y_data[min_idx,:].reshape((1,-1))

def get_sub_idx_from_unordered_set(K,n_sel,rand_rate=0.0):
    n_total = K.shape[0]
    remain_idxs = np.arange(n_total)
    sub_idx = np.zeros((n_sel))
    sum_K_vec = np.zeros(n_total)
    for i_idx in range(n_sel):
        if i_idx == 0:
            sel_idx = np.random.randint(n_total)
        else:
            curr_K_vec = K[(int)(sub_idx[i_idx-1]),:] 
            sum_K_vec = sum_K_vec + curr_K_vec
            k_vals = sum_K_vec[remain_idxs]
            min_idx = np.argmin(k_vals)
            sel_idx = remain_idxs[min_idx] 
            if rand_rate > np.random.rand():
                rand_idx = np.random.choice(len(remain_idxs),1,replace=False)  
                sel_idx = remain_idxs[rand_idx] 
        sub_idx[i_idx] = (int)(sel_idx)
        remain_idxs = np.delete(remain_idxs,np.argwhere(remain_idxs==sel_idx))
    sub_idx = sub_idx.astype(np.int) # make it int
    return sub_idx

def get_x_sub_kdpp(x_minmax,n_sel,n_raw=10000,invlen=100):
    x_raw = np.asarray(x_sampler(n_sample=n_raw,x_minmax=x_minmax))[:,0,:]
    K = k_m52(x1=x_raw,x2=x_raw,x_range=x_minmax[:,1]-x_minmax[:,0],invlen=invlen) 
    sub_idx = get_sub_idx_from_unordered_set(K,n_sel=n_sel)
    x_sub = x_raw[sub_idx,:]
    return x_sub

def argmax_acquisition(x_data,y_data,x_minmax,
                       n_retry=10,kgain=1.0,invlen=10,eps=1e-6,maxiter=100,
                       n_cd_step=0,n_cd_resolution=0):
    """
    Get the maxima of the acquisition function from multiple random restarts 
    """
    xdim = x_minmax.shape[0]
    # First, compuate GP statistics
    xmin,xmax = x_minmax[:,0],x_minmax[:,1]
    x_range = xmax-xmin
    K = k_m52(x1=x_data,x2=x_data,x_range=x_range,gain=kgain,invlen=invlen) # kernel matrix
    inv_K = np.linalg.inv(K+eps*np.eye(x_data.shape[0]))
    def a_test(x_test,SCALE_Y=True):
        """
        Acquisition function with precomputed invK
        """
        if SCALE_Y:
            y_data_copy = np.copy(y_data)
            y_data_mean = np.mean(y_data_copy)
            y_data_mz = y_data_copy - y_data_mean
            y_min,y_max = np.min(y_data_mz), np.max(y_data_mz)
            y_range = y_max - y_min
            y_data_scaled = 2.0 * y_data_mz / y_range
        else:
            y_data_scaled = np.copy(y_data)
        # Get GP mu and var
        k_test = k_m52(x1=x_test,x2=x_data,
                       x_range=x_range,gain=kgain,invlen=invlen)
        W = np.matmul(k_test,inv_K)
        mu_y = np.mean(y_data_scaled)
        mu_test = np.matmul(W,y_data_scaled-mu_y)+mu_y
        var_test = (kgain-np.diag(np.matmul(W,k_test.T))).reshape((-1,1))
        # Expected improvement criterion 
        gamma = (np.min(y_data_scaled) - mu_test)/sqrt_safe(var_test)
        a_ei = 2.0 * sqrt_safe(var_test) * (gamma*Phi(gamma) + norm.pdf(mu_test,0,1))
        if SCALE_Y:
            mu_test = 0.5 * y_range * mu_test + y_data_mean
        return a_ei,mu_test,var_test
    def f_temp(x):
        """
        Cost function to be minimized = -acquisition function
        """
        a_ei,mu_test,var_test = a_test(x.reshape((1,-1)))
        return -a_ei # flip as we will find minima
    # Unifromly find initial inputs using k-DPP
    x_inits = get_x_sub_kdpp(x_minmax,n_sel=n_retry,n_raw=5000,invlen=100)
    # Skip half of them to start near 'x_best'
    x_bset,_ = get_best_xy(x_data,y_data)
    x_inits[n_retry//2:,:] = \
        x_bset + 0.1*sqrt_safe(x_minmax[:,1]-x_minmax[:,0]).reshape((1,-1))*np.random.randn(n_retry//2,xdim)
    
    # Get the maxima of the acquisition function from multiple initial points
    x_opts,f_opts,f_inits = np.zeros((n_retry,xdim)),np.zeros(n_retry),np.zeros(n_retry)
    for i_idx,x_init in enumerate(x_inits): # multiple retry 
        f_inits[i_idx] = f_temp(x_init)
        x_opt = scipy.optimize.fmin(func=f_temp,x0=x_init,disp=0,maxiter=maxiter) # optimized input [1 x d]
        x_opt = np.maximum(np.minimum(x_opt,xmax),xmin) # clip values with x_minmax
        x_opt = x_opt.reshape((1,-1)) # rank 2 
        f_opt = f_temp(x_opt)
        x_opts[i_idx,:],f_opts[i_idx] = x_opt,f_opt    
    x_best = x_opts[np.argmin(f_opts),:]
    # Fine-tuning with coordinate descent
    if n_cd_step > 0:
        for _ in range(n_cd_step):
            for d_idx in range(xdim): # for different dim
                x_best_copy = np.copy(x_best) # copy
                xmin_d,xmax_d = xmin[d_idx],xmax[d_idx] # range 
                f_vals,x_d_vals = np.zeros(n_cd_resolution),np.linspace(xmin_d,xmax_d,n_cd_resolution)
                for i_idx,x_d in enumerate(x_d_vals):
                    if i_idx == 0: # reserve the first one to be the current solution 
                        x_best_copy[d_idx] = x_best[d_idx]  
                    else:
                        x_best_copy[d_idx] = x_d 
                    f_vals[i_idx] = f_temp(x_best_copy) # find minima
                x_best[d_idx] = x_d_vals[np.argmin(f_vals)] # update
    return x_best.reshape((1,-1)) 
    