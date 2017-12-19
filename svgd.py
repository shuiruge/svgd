from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

def euc_dist(tX, X):
    k1 = tf.reshape(tf.reduce_sum(tf.square(tX),axis=1),[-1,1])
    k2 = tf.tile(tf.reshape(tf.reduce_sum(tf.square(X),axis=1),[1,-1]),[tX.shape[0].value,1])
    k = k1+k2-2*tf.matmul(tX,tf.transpose(X))

    return k

def rbf_kernel(pairwise_dist, h=1.):
    
    return tf.exp(-1.*pairwise_dist/h)

def get_median(v):    
    v = tf.reshape(v, [-1])
    m = v.shape[0].value//2
    return tf.nn.top_k(v, m).values[m-1]


class SVGD:
    def __init__(self,h=None):       
        self.h = h

    def svgd_kernel(self, X):
        
        pdist = euc_dist(X,X)

        if self.h is None:
            if X.shape[0].value == 1:
                h = 1.
            else:
                h = get_median(pdist)  
            self.h = tf.sqrt(0.5 * h / tf.log(X.shape[0].value+1.))

        kxy = rbf_kernel(pdist,self.h)

        dx = tf.expand_dims(X,[1]) - tf.expand_dims(X,[0])
        dkxy = 2*tf.matmul(tf.expand_dims(kxy,[1]),dx)/self.h

        return kxy, tf.squeeze(dkxy)

    def gradient(self,X,dlnprob):
        N = X.shape[0].value
        kxy, dkxy = self.svgd_kernel(X)
  
        grad = -(tf.matmul(kxy, dlnprob) + dkxy)/N

        return grad

    def update(self,X,dlnprob,vars,niter=1000,optimizer=None):

        sgrad = self.gradient(X,dlnprob)

        if optimizer is None:
            global_step = tf.Variable(0, trainable=False, name="global_step")
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                                    global_step,
                                                                    100, 0.9, staircase=True)
            #from the implementation of the author
            optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.,beta2=0.9)

        train = optimizer.apply_gradients([(sgrad,vars)],global_step=global_step)

        sess = ed.get_session()
        tf.global_variables_initializer().run()

        iter_num = 1000
        for _ in range(iter_num):
            #print('sgrad',sess.run(sgrad))
            sess.run(train)