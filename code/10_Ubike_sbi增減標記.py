#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
os.chdir('D:/project/專題/資料庫備份')


# In[ ]:


data_8 = pd.read_csv('缺車率計算.csv')
data_9 = pd.read_csv('bike_9_all.csv')


# In[ ]:


# data_8 = data_8[['sno','sna','sbi','sarea','mday','lat','lng','bemp','weekday']]
# data_8.head()


# # In[ ]:


# data_9 = data_9[['sno','sna','sbi','sarea','mday','lat','lng','bemp','weekday']]
# data_9.tail()


# In[ ]:


# data = pd.concat([data_8,data_9])
data = pd.read_csv('尖離峰min_v2.csv')


# In[ ]:


# data.index = data['sno']


# In[ ]:


sno = data.groupby('sno')


# In[ ]:


sno.size()


# In[ ]:


sno_list = list(sno.size().index)
sno_list



# In[ ]:


def get_group(sno, sno_n):
    group = sno.get_group(sno_n)
    mydict = group.to_dict('records')
    for i in range(len(mydict)):
        if i == 0:
            mydict[i]['sbi_I/D'] = 0
        else:
            mydict[i]['sbi_I/D'] = mydict[i]['sbi']-mydict[i-1]['sbi']
    return mydict


# In[ ]:


for sno_n in sno_list[199:]:
    if sno_n == 1:
        mydict = get_group(sno, sno_n)
        df = mydict
    else:
        mydict = get_group(sno, sno_n)
#         df_new = pd.DataFrame.from_dict(mydict)
        df = df+mydict
    # print(sno_n)


# In[ ]:

# 會當掉，分批跑
df_new1 = pd.DataFrame.from_dict(df[0:8292425])
df_new2 = pd.DataFrame.from_dict(df[8292425:16584850])
df_new3 = pd.DataFrame.from_dict(df[16584850:24877275])
df_new4 = pd.DataFrame.from_dict(df[24877275:])
# df_new5 = pd.DataFrame.from_dict(df[30000000:])


# In[ ]:


df_new = pd.concat([df_new1, df_new2])
df_new = pd.concat([df_new, df_new3])
df_new = pd.concat([df_new, df_new4])


# In[ ]:


df_new.head()


# In[ ]:


df_new.tail()


# In[ ]:


df_new.reset_index(drop=True, inplace=True)


# In[ ]:


df_new.to_csv('尖離峰min_v2.csv', index = False)

