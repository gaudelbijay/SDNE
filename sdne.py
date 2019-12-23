import time 
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 
import tensorflow as tf 
from tensorflow.keras import backend as k 
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

def L_1st(alpha):
    def loss_1st(y_true,y_pred):
        L = y_true 
        Y = y_pred
        batch_size = tf.to_float(k.shape(L)[0])
        return alpha*2*tf.linalg.trace(tf.matmul(tf.matmul(Y,L,transpose_a=True),y))/batch_size
    return loss_1st

def L_2nd(beta):
    def loss_2nd(y_true,y_pred):
        B = np.ones_like(y_true)
        B[y_true != 0] = beta 
        x = k.square((y_true-y_pred)*B)
        t = k.sum(x,axis=-1)
        return k.mean(t)
    return loss_2nd