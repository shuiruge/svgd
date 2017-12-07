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

def rbf_kernel(tX, X, h=1.):
    dist = euc_dist(tX,X)
    return tf.exp(-1.*dist/h)

class SVGD:
    def __init__(self, X, dlnprob, h=1.):
        self.X = X
        self.dlnprob = dlnprob
        self.h = h

    def svgd_kernel(self, X, h):
        kxy = rbf_kernel(X,X,h)

        dx = tf.expand_dims(X,[1]) - tf.expand_dims(X,[0])
        dkxy = 2*tf.matmul(tf.expand_dims(kxy,[1]),dx)/h

        return kxy, tf.squeeze(dkxy)

    def gradient(self):
        N = self.X.shape[0].value
        kxy, dkxy = self.svgd_kernel(self.X, self.h)
        grad = (tf.matmul(kxy, self.dlnprob(self.X)) + dkxy)/N

        return grad