# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:05:29 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")

data = pd.read_csv('89月捕的.csv')
group = data.groupby('sno')
for i in range(1,11):
    if i == 1:
        g = group.get_group(i)
    else:
        q = group.get_group(i)
        g = pd.concat([g, q])
        
g.reset_index(drop =True, inplace = True)
g.to_csv('前10站89月.csv', index = False)
