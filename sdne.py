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