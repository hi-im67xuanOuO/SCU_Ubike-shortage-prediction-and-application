# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:34:59 2020

@author: s9631
"""
import os
os.chdir('D:/project/專題/資料庫備份/合併天氣資料_min')
# os.chdir('D:/project/專題/資料庫備份/前10站89月')

import pandas as pd
import numpy as np
import random

from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from keras import optimizers
df = pd.read_csv('outlier移除後.csv')

X = df[['sno', 'tot', 'weekday', 'sbi_I/D', 'Precp', 'RH', 'Temperature', 'hour', '中山區',
       '中正區', '信義區', '內湖區', '北投區', '南港區', '士林區', '大同區', '大安區', '文山區', '松山區',
       '萬華區','peak']].values
y = df[['sbi_tot_mean']]
z = df[['mday']]

from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)
X_mm = pd.DataFrame(X_mm)
df2 = pd.concat([X_mm, y, z], axis=1)
df2.columns = ['sno', 'tot', 'weekday', 'sbi_I/D', 'Precp', 'RH', 'Temperature', 'hour', '中山區',
       '中正區', '信義區', '內湖區', '北投區', '南港區', '士林區', '大同區', '大安區', '文山區', '松山區',
       '萬華區', 'peak', 'sbi_tot_mean', 'mday']
df2.head()

df_sorted = df2.sort_values(by=['sno', 'mday'], ascending = [True, True]).reset_index(drop=True)
df_train = df_sorted[df_sorted['mday'] < '2020-09-19 00:00:00'].reset_index()
df_test = df_sorted[df_sorted['mday'] >= '2020-09-18 00:50:00'].reset_index()
# df_train = df_sorted[df_sorted['mday'] < '2020-08-22 00:00:00'].reset_index()
# df_test = df_sorted[df_sorted['mday'] >= '2020-08-21 00:50:00'].reset_index()

df_train['rank'] = df_train.groupby(['sno'])['index'].rank().astype(int)
df_test['rank'] = df_test.groupby(['sno'])['index'].rank().astype(int)

# print(len(df_train))
# print(len(df_test))
def process_data(df_train, df_test, pastDay, futureDay):
    feature = ['tot', 'weekday', 'peak', 'sbi_I/D', 'Precp', 'RH', 'Temperature', '中山區',
       '中正區', '信義區', '內湖區', '北投區', '南港區', '士林區', '大同區', '大安區', '文山區', '松山區',
       '萬華區']
    target = 'sbi_tot_mean'
    train_feature = np.array(df_train[feature])
    train_flow = np.array(df_train[target])
    train_rank = np.array(df_train['rank'])
    
    test_feature = np.array(df_test[feature])
    test_flow = np.array(df_test[target])
    test_rank = np.array(df_test['rank'])
    
    x_train, x_test, y_train, y_test = [],[],[],[]
    
    for i in tqdm(range(pastDay, len(train_feature))):
        if train_rank[i] > pastDay:#每一站的rank=1~10要刪掉
            x_train.append(train_feature[i - pastDay: i])
            y_train.append(train_flow[i: i + futureDay])
        else:
            pass
        
    for i in tqdm(range(pastDay, len(test_feature))):
        if test_rank[i] > pastDay:
            x_test.append(test_feature[i - pastDay: i])
            y_test.append(test_flow[i: i + futureDay])
        else:
            pass
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # random_index = list(range(len(x_train)))
    # random.shuffle(random_index)
    # x_train = x_train[random_index]
    # y_train = y_train[random_index]
    
    print('x_train_shape:', x_train.shape)
    print('y_train_shape:', y_train.shape)
    print('x_test_shape:', x_test.shape)
    print('y_test_shape:', y_test.shape)
    
    return x_train, y_train, x_test, y_test

pastDay = 10
futureDay = 1#可以試試看增加，看誤差多少

x_train, y_train, x_test, y_test = process_data(df_train, df_test, pastDay, futureDay)

# 建立模型
def get_lstm(x_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_shape[1], x_shape[2])))
    # model.add(Dropout(0,2))
    model.add(Dense(1, activation='sigmoid'))# 激活(?
    optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    return model

model = get_lstm(x_train.shape)

from keras import backend as K

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
# 激活(?
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_error', 'mean_squared_error', r2_score])

print(model.summary())

# validation篩選模型最好的參數(?
# epochs訓練幾次
hist = model.fit(x_train, y_train, batch_size=300, epochs=10, validation_split=0.3)
# valisation_data = (x_val, y_val)

hist.history.keys()

def Saveplot(history):
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.plot(history.epoch, history.history['loss'], 'red', lw=2, label='Training')
    plt.plot(history.epoch, history.history['val_loss'], 'blue', lw=2, label='Validation')
    plt.legend()
    plt.show()
    
def evaluation(y_test, predictions):
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    
    print('Accuracy:', str(round((tp+tn)/(tp+fp+fn+tn),3)))
    print('Recall:', str(round((tp)/(tp+fn), 3)))
    print('Precision:', str(round((tp)/(tp+fp), 3)))
    print('f1_score:', str(f1_score(y_test, predictions, zero_division=1)))
    print('-'*30)
    print(confusion_matrix(y_test, predictions).ravel())
    print('tn:', str(round(tn,3)), '    ', 'fp:', str(round(fp,3)))
    print('fn:', str(round(fn,3)), '    ', 'tp:', str(round(tp,3)))
    
prediction = model.predict(x_test)
prediction_class = model.predict_classes(x_test)

evaluation(y_test, prediction_class)

model.save( 'lstm_model_v2.h5' )

Saveplot(hist)

#匯入上次跑完的模型
from keras.models import load_model

model = load_model( 'lstm_model.h5' )

from sklearn.metrics import r2_score
print('R^2:',r2_score(y_test, prediction))

# def Saveplot_MAE(history):
#     plt.xlabel('Number of Epochs')
#     plt.ylabel('MAE')
#     plt.plot(history.epoch, history.history['mean_absolute_error'], 'green', lw=2, label='Training')
#     plt.plot(history.epoch, history.history['val_mean_absolute_error'], 'orange', lw=2, label='Validation')
#     plt.legend()
#     plt.show()
    
# Saveplot_MAE(hist)

def combine_test(df_test):
    test_group = df_test.groupby('sno')
    sno_list = list(test_group.size().index)
    for s in tqdm(sno_list):
        test = test_group.get_group(s)
        test.reset_index(drop=True, inplace=True)
        if s == 1:
            t = test[10:]
        else:
            t = pd.concat([t, test[10:]])
    return t
t1 = combine_test(df_test)
t1_sorted = t1.sort_values(by=['sno', 'mday'], ascending = [True, True]).reset_index(drop=True)
t1_sorted['predict_400'], t1_sorted['predict_600'], t1_sorted['predict_1000'] = prediction3, prediction, prediction2
t1_sorted['MSE600'] = t1_sorted['']