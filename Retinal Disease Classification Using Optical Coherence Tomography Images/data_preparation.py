#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:08:21 2024

@author: venkatesh
"""

#%%
# data making

import os
print(os.listdir('./OCT_data/train/NORMAL'))
print(os.listdir('./OCT_data/train/DRUSEN'))
print(os.listdir('./OCT_data/train/DME'))
print(os.listdir('./OCT_data/train/CNV'))


#%%
paths=[]
labels=[]
train_path='./OCT_data/train/'
for i in os.listdir(train_path):
    for j in os.listdir(train_path+"/"+i):
        paths.append(train_path+"/"+i+"/"+j)
        labels.append(i)
# saving the model losses

import pandas as pd

loss_dic={"image_path":paths,
          "label":labels}

df = pd.DataFrame.from_dict(loss_dic) 
path='csv_files/data.csv'
df.to_csv (path, index = False, header=True)
#%%

# data making

import os
print(os.listdir('./OCT_data/test/NORMAL'))
print(os.listdir('./OCT_data/test/DRUSEN'))
print(os.listdir('./OCT_data/test/DME'))
print(os.listdir('./OCT_data/test/CNV'))



paths=[]
labels=[]
train_path='./OCT_data/test/'
for i in os.listdir(train_path):
    for j in os.listdir(train_path+"/"+i):
        paths.append(train_path+"/"+i+"/"+j)
        labels.append(i)
# saving the model losses

import pandas as pd

loss_dic={"image_path":paths,
          "label":labels}

df = pd.DataFrame.from_dict(loss_dic) 
path='csv_files/full_test_data.csv'
df.to_csv (path, index = False, header=True)