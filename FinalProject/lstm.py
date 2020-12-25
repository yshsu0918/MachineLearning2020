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


import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data

import pickle #pickle模块

from func import *



class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,          # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, _ = self.rnn(x, None)   # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])
        return out
def main_pytorch(X_train, Y_train, X_val, Y_val):

    torch_traindataset = Data.TensorDataset(torch.tensor(X_train).float() ,torch.tensor(Y_train).float() )
    # 把 dataset 放入 DataLoader
    train_loader  = Data.DataLoader(
        dataset=torch_traindataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )

    torch_testdataset = Data.TensorDataset(torch.tensor(X_val).float() ,torch.tensor(Y_val).float() )
    test_loader  = Data.DataLoader(
        dataset=torch_testdataset,      # torch TensorDataset format
        batch_size=1,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )


    rnn = RNN().cuda()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.MSELoss()                       

    # training and testing
    #rnn = rnn.float()
    
    for epoch in range(EPOCH):
        avg_loss = 0
        totalstep = 0
        for step, (b_x, b_y) in enumerate(train_loader):    # gives batch data
            #print('b_x',b_x.shape)
            output = rnn(b_x.cuda())                                # rnn output
            #print('rnn output', output.shape)
            #print(output)
            #print('b_y', b_y.shape)
            loss = loss_func(output, b_y.cuda())                   # MSE loss
            #print(loss)
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            totalstep+= 1
            avg_loss += loss
        avg_loss/=totalstep
        
        
            
        with torch.no_grad():
            TEST_QuoteChange_AVG = 0
            totalstep = 0
            for teststep, (test_x, test_y) in enumerate(test_loader):
                output = rnn(test_x.cuda())
                # print('IN TEST')
                # print('rnn output', output.shape)
                # 
                output_ = output.cpu()
                if output_ > 0.1:
                    output_ = 0.1
                if output < -0.1:
                    output_ = -0.1
                
                #print('target_y {} pred_y {} delta {}'.format(test_y, output_ , abs(test_y-output_)))
                
                TEST_QuoteChange_AVG += abs(test_y-output_)
                totalstep += 1 
            TEST_QuoteChange_AVG /= totalstep
            print('EPOCH {} TEST_QuoteChange_AVG {} avg_loss {}'.format(epoch, TEST_QuoteChange_AVG, avg_loss))



def main_keras(X_train, Y_train, X_val, Y_val, loss_figure_fname):
    # train = readTrain()
    # #train_Aug = augFeatures(train)
    # train_norm = normalize(train)
    # # change the last day and next day 
    # X_train, Y_train = buildTrain(train_norm, 30, 1)
    # X_train, Y_train = shuffle(X_train, Y_train)
    # # because no return sequence, Y_train and Y_val shape must be 2 dimension
    # X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

    model = buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])

    Y_predict = model.predict(X_val)
    #print( Y_predict, Y_val)
    avg = 0
    amount_of_test = 0
    for yv,yp in zip( Y_predict.reshape(-1).tolist(), Y_val.reshape(-1).tolist(), ):
        if yp > 0.1:
            yp = 0.1
        if yp < -0.1:
            yp = -0.1
        #print(yv,yp, abs(yv-yp))
        avg += abs(yp - yv)
        amount_of_test += 1
    avg /= amount_of_test
    print( 'avg ', avg)
    
    plt.cla()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.savefig(loss_figure_fname)


    return model
    #-------------------------------------------------






torch.manual_seed(1)    # reproducible
EPOCH = 1000               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 20          # rnn time step
INPUT_SIZE = 10         # rnn input size
LR = 0.01               # learning rate

Days_before = 10

#stocklist = ['00632R.TW','00633L.TW','00633L.TW','00633L.TW','00633L.TW','00634R.TW','00637L.TW','00637L.TW','00637L.TW','00637L.TW','00672L.TW','00672L.TW','00672L.TW','00672L.TW','1218.TW','1312.TW','1455.TW','1512.TW','1515.TW','1536.TW','1605.TW','1909.TW','2308.TW','2337.TW','2337.TW','2344.TW','2345.TW','2353.TW','2356.TW','2371.TW','2377.TW','2382.TW','2383.TW','2383.TW','2406.TW','2406.TW','2406.TW','2408.TW','2409.TW','2409.TW','2439.TW','2448.TW','2474.TW','2474.TW','2474.TW','2492.TW','2492.TW','2492.TW','2492.TW','2498.TW','2498.TW','2548.TW','2603.TW','2609.TW','2609.TW','2633.TW','2885.TW','2903.TW','3005.TW','3005.TW','3006.TW','3006.TW','3016.TW','3016.TW','3018.TW','3019.TW','3034.TW','3037.TW','3041.TW','3049.TW','3167.TW','3167.TW','3189.TW','3189.TW','3231.TW','3231.TW','3311.TW','3311.TW','3380.TW','3406.TW','3406.TW','3413.TW','3413.TW','3443.TW','3443.TW','3443.TW','3450.TW','3450.TW','3450.TW','3481.TW','3545.TW','3545.TW','3576.TW','3673.TW','4938.TW','4968.TW','4968.TW','4968.TW','4977.TW','5264.TW','5264.TW','6285.TW','6456.TW','6456.TW','6456.TW','8105.TW','8112.TW','8150.TW','8404.TW','9802.TW','9802.TW','9904.TW','9910.TW','9910.TW']

stocklist = ['2330.TW','2880.TW']





if __name__ == '__main__': 

    output_model_dir = './kerasmodel_save'
    if not os.path.isdir(output_model_dir):
        os.mkdir(output_model_dir)
    output_lossfig_dir = './kerasmodel_lossfig'
    if not os.path.isdir(output_lossfig_dir):
        os.mkdir(output_lossfig_dir)

    for stocknum in stocklist:

        historical_data = os.path.join('./20clean',stocknum.replace('.TW','.TW.csv'))

        train = readTrain(historical_data)
        train_Aug = augFeatures(train)
        
        _, Y_train = buildTrain(train_Aug, Days_before, 1)
        train_norm = normalize(train_Aug)
        # change the last day and next day 
        X_train, _ = buildTrain(train_norm, Days_before, 1)
        X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
        X_train, Y_train = shuffle(X_train, Y_train)
        
        model = main_keras(X_train, Y_train, X_val, Y_val, os.path.join(output_lossfig_dir, stocknum.replace('.TW','.TW.png')))

        modelfn = os.path.join(output_model_dir, stocknum.replace('.TW', ''))
        model.save(modelfn)


    
    
    
    #main_pytorch(X_train, Y_train, X_val, Y_val)


