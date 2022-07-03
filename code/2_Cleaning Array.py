# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:25:36 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")

data = pd.read_csv('b1.csv')
name = ['sno','sna','tot','sbi','sarea','mday','lat','lng','ar','bemp','act']
c1 = list(data.columns)
c1 = pd.DataFrame(c1)
c1 = c1.T
c1 = c1[[0,1,2,3,4,5,6,7,8,12,13]]
c1.columns = name

data.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
data = data[[0,1,2,3,4,5,6,7,8,12,13]]
data.columns = name

data = pd.concat([c1, data]).reset_index(drop = True)
data.to_csv('b1.csv', index = False)

del c1,name

data_copy = data[['sno', 'sna', 'lat', 'lng']]
data_copy.to_csv('array.csv', index = False)
