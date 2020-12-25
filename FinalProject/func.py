# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data

import pickle #pickle模块
def readTrain(fn = '2330.TW_2000.csv', end_date = '2016-12-31'):
    train = pd.read_csv(fn)
    return train
  
def augFeatures(train):
    train["Date"] = pd.to_datetime(train["Date"])
    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month
    train["date"] = train["Date"].dt.day
    train["day"] = train["Date"].dt.dayofweek
    return train

def normalize(train):
    train = train.drop(["Date"], axis=1)
    #print('###drop train @normalize')
    #peep(train)
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    #train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) ))

    return train_norm

def buildTrain(train, pastDay, futureDay):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))

        day = train.iloc[i+pastDay]["Adj Close"]
        dayplus1 = train.iloc[i+pastDay+1]["Adj Close"]
        
        QuoteChange = np.array( [ (dayplus1-day)/day ]  )
        #print(day, dayplus1, QuoteChange)
        Y_train.append(QuoteChange)

    return np.array(X_train), np.array(Y_train)

def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val

def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

def peep(lst,num=5):
    
    for x in lst[:num]:
        pass
        #print(x)


def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model



def lstm_model2Ypredict(modelpath , modelfn):
    Days_before = 10
    print(modelpath, modelfn)
    model = load_model(modelpath)
    
    testdata = readTrain(os.path.join('./20clean_test',modelfn+'.TW.csv'))
    testdata_Aug = augFeatures(testdata)
    _, Y_val = buildTrain(testdata_Aug, Days_before, 1)
    test_norm = normalize(testdata_Aug)
    X_val, _ = buildTrain(test_norm, Days_before, 1)
    


    Y_predict = model.predict(X_val)

    return Y_predict
