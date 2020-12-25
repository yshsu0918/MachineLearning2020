# -*- coding: UTF-8 -*-
import os
import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from func import *



Days_before = 10
if __name__ == '__main__': 



    input_model_dir = './kerasmodel_save'
    
    for modelfn in os.listdir(input_model_dir):
        if '2330' not in modelfn:
            continue
        print(modelfn)
        os.path.join(input_model_dir, modelfn)

    
