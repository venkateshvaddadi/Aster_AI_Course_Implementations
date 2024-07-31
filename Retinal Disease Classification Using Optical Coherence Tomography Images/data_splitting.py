#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:39:55 2024

@author: venkatesh
"""

import os
os.listdir('./csv_files/')
#%%

import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Read the original CSV file
data = pd.read_csv('./csv_files/data.csv')
#%%
# Step 2: Split the data into training (80%) and remaining data (20%)
# train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)


# # Step 4: Save the training, validation, and testing sets to separate CSV files
# train_data.to_csv('./csv_files/data_train.csv', index=False)
# valid_data.to_csv('./csv_files/data_valid.csv', index=False)

# print("Data split and saved successfully!")
#%%
# from the given 1 Laksh data points we are taking 20000 thousand points for the experiment.
train_data, remaining_data = train_test_split(data, test_size=0.8, random_state=42)

# splitting the data.
train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)
# Step 3: Split the remaining data into validation (10%) and testing (10%)
test_data, valid_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Step 4: Save the training, validation, and testing sets to separate CSV files
train_data.to_csv('./csv_files/data_train.csv', index=False)
valid_data.to_csv('./csv_files/data_valid.csv', index=False)
test_data.to_csv('./csv_files/data_test.csv', index=False)

print("Data split and saved successfully!")
#%%

import pandas as pd
print('traing data')
data = pd.read_csv('./csv_files/data_train.csv')

print(data.head())
# Print the statastis of the given data
print("stastics of the given data:")
print(data['label'].value_counts())
print('#'*100)

print('validation data')

data = pd.read_csv('./csv_files/data_valid.csv')

print(data.head())
# Print the statastis of the given data
print("stastics of the given data:")
print(data['label'].value_counts())
print('#'*100)

print('test data')

data = pd.read_csv('./csv_files/data_test.csv')

print(data.head())
# Print the statastis of the given data
print("stastics of the given data:")
print(data['label'].value_counts())
print('#'*100)
#%%