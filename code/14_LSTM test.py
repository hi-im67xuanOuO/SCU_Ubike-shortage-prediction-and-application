#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random

from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm_notebook


# In[2]:
import os
os.chdir('D:/project/專題/資料庫備份/合併天氣資料_min')


df = pd.read_csv('8月.csv')#, encoding = 'big5')
df.head()


# In[3]:


# df.columns


# In[4]:


# new_df = df.drop(['最近車站', 'sna_y','type','bemp','sbi','sbi_tot','hour','new_Date','INDEX','Unnamed: 0'], axis=1)
# new_df


# In[5]:


df.columns


# In[6]:


df_train = df[df['mday']<'2020-08-22 00:00:00'].reset_index()
df_test = df[df['mday']>='2020-08-21 00:50:00'].reset_index()


# In[7]:


df_train.head()


# In[8]:


df_test.tail()


# In[9]:


df_train['peak'].value_counts() #類別用改成peak 數值型態用sbi_tot_mean


# In[10]:


df_test['peak'].value_counts()


# In[11]:


df_train['rank'] = df_train.groupby(['sno'])['index'].rank().astype(int)
df_test['rank'] = df_test.groupby(['sno'])['index'].rank().astype(int)


# In[13]:


#類別target = peak 數值 target = sbi_tot_mean
def process_data(df_train, df_test, pastDay, futureDay):
    feature = ['tot','sbi_tot_mean','sbi_I/D','weekday','RH','Precp','Temperature'] #改sarea共12區、天氣3個 
    
    target = 'peak'
    
    train_feature = np.array(df_train[feature])
    train_flow = np.array(df_train[target])
    train_rank = np.array(df_train['rank'])
    
    test_feature = np.array(df_test[feature])
    test_flow = np.array(df_test[target])
    test_rank = np.array(df_test['rank'])
    
    X_train, y_train, X_test, y_test = [],[],[],[]
    
    for i in tqdm_notebook(range(pastDay, len(train_feature))):
        if train_rank[i] > pastDay:
            X_train.append(train_feature[i - pastDay: i])
            y_train.append(train_flow[i:i+futureDay])
        else:
            pass
        
    for i in tqdm_notebook(range(pastDay, len(test_feature))):
        if test_rank[i] > pastDay:
            X_test.append(test_feature[i - pastDay: i])
            y_test.append(test_flow[i:i+futureDay])
        else:
            pass
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
#     random_index = list(range(len(X_train)))
#     random.shuffle(random_index)
#     X_train = X_train[random_index]
#     y_train = y_train[ramdom_index]
    
    print('X_train_shape:', X_train.shape)
    print('y_train_shape:', y_train.shape)
    print('X_test_shape:', X_test.shape)
    print('X_test_shape:', y_test.shape)
    
    return X_train, y_train, X_test, y_test


# In[14]:


pastDay = 10
futureDay = 1
del df


# In[15]:


X_train, y_train, X_test, y_test = process_data(df_train, df_test, pastDay, futureDay)


# In[16]:


X_train[180]


# In[17]:


def get_lstm(X_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape = (X_shape[1], X_shape[2])))
    model.add(Dense(1, activation = "sigmoid"))
    
    return model


# In[18]:


model = get_lstm(X_train.shape)
#model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
#model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_error'])
model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())


# In[19]:


hist = model.fit(X_train, y_train, batch_size = 300, epochs = 10, validation_split = 0.05)


# In[20]:


hist.history.keys() #dict裡面的key


# In[ ]:





# In[21]:


def Saveplot(history):
    plt.xlabel('Number of Epochs')
    #plt.xlabel('Accuracy') 跑accuracy再用
    plt.ylabel('Loss')
    plt.plot(history.epoch, history.history['loss'], 'red', lw=2, label = 'Training')
    plt.plot(history.epoch, history.history['val_loss'], 'blue', lw=2, label = 'Validation')
    
    plt.legend()
    plt.show()


# In[22]:


Saveplot(hist)


# In[ ]:





# In[71]:


def evaluation(y_test, predictions):
    tn, fp, fn, tp = confusion_matrix(y_test.ravel().tolist(), predictions.ravel().tolist()).ravel()
    
    print("Accuracy: "+str(round((tp+tn)/(tp+fp+fn+tn), 3)))
    print("Recall: "+str(round((tp)/(tp+fn), 3)))
    print("Precision: "+str(round((tp)/(tp+fp), 3)))
#     print("f1_score: "+str(f1_score(y_test, predictions, zero_devision=1)))
    print('-'*30)
    print(confusion_matrix(y_test.ravel(), predictions.ravel()).ravel())
    print("tn: "+str(round(tn ,3)),'  ','fp: '+str(round(fp, 3)))
    print("fn: "+str(round(fn ,3)),'  ','tp: '+str(round(tp, 3)))


# In[42]:


prediction = model.predict(X_test)
prediction


# In[43]:


prediction_class = model.predict_classes(X_test)
prediction_class


# In[44]:


# predict = pd.DataFrame(prediction)
# predict_class = pd.DataFrame(prediction_class)
# predict.head()


# In[45]:


# df_two = pd.concat([predict, predict_class], axis = 1)
# df_last = pd.concat([df_test, df_two], axis = 1)
# df_last.head()


# In[46]:


# df_last.tail()


# In[47]:


# df_last.to_csv('LSTM類別型預測結果.csv')


# In[48]:


#儲存模型
model.save( 'lstm_model.h5' )


# In[49]:


#匯入上次跑完的模型
from keras.models import load_model

model = load_model( 'lstm_model.h5' )


# In[72]:


evaluation(y_test,prediction_class)


# In[65]:


print(y_test)
print(type(y_test))


# In[66]:


ytest = y_test.tolist()


# In[68]:


print(prediction_class)
print(type(prediction_class))


# In[69]:


predictionclass = prediction_class.tolist()


# In[70]:


evaluation(ytest,predictionclass)


# In[ ]:





# In[ ]:





# In[62]:


test = prediction_class.ravel()
test = test.tolist()


# In[57]:


type(test)


# In[ ]:




