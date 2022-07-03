# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")
def data_clean(name,output_name):
    data = pd.read_csv(name+".csv")
    all_data = len(data)
    print(all_data)
    first = list(data.columns)
    col_name = ['sno','sna','tot','sbi','sarea','mday','lat','lng','ar','sareaen','snaen','aren','bemp','act']
    df = pd.DataFrame(first).T
    df.columns = col_name
    data.columns = col_name
    data = pd.concat([df, data]).reset_index(drop = True)
    data.to_csv(output_name+'.csv', index = False)
    return data
    
data_clean("bike1", "bike_8_ok")#8月
data = data_clean("bike_9", "bike_9_ok")#9月
