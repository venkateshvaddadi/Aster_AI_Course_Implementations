#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:43:57 2024

@author: venkatesh
"""

import os
print(os.listdir('./processed_data/pocus_images/'))


#%%
data_path='./processed_data/'
image_path=data_path+'/pocus_images/'

#%%

image_paths=[]
image_labels=[]

for i in os.listdir(image_path):
    for label in os.listdir(image_path+'/'+i):
        print(label)
        for image in os.listdir(image_path+'/'+i+'/'+label):
            print(image_path+'/'+i+'/'+label+'/'+image,label)
            image_paths.append(image_path+'/'+i+'/'+label+'/'+image)
            image_labels.append(label)
            
            
            
#%%
videos_path=data_path+'/pocus_videos/'
print(os.listdir(videos_path))
import cv2

for i in os.listdir(videos_path):
    for label in os.listdir(videos_path+'/'+i):
        # print(label)
        for video_name in os.listdir(videos_path+'/'+i+'/'+label):
            video_path=videos_path+'/'+i+'/'+label+'/'+video_name
            video_label=label
            print(video_path,video_label)
            # need to write a code here to convert the video into images
            if(video_name.lower().endswith('.avi') or video_name.lower().endswith('.mp4') or video_name.lower().endswith('.mov')):
                print(video_name)

                video_to_folder_name = os.path.splitext(os.path.basename(video_name))[0]
                # print(video_name)
                try:
                    video_to_folder_full_path=data_path+'pocus_videos_to_images/'+i+'/'+label+'/'+video_to_folder_name
                    os.makedirs(video_to_folder_full_path)
                except:
                    print('folder already existed')
                    # code for converting video into images
                video = cv2.VideoCapture(video_path)
                frame_count = 0
                # Read frames from the video and save them as images
                ret=True
                while ret:
                    ret, frame = video.read()
                    if not ret:
                        break
                    print(frame.shape)

                    # Save the frame as a PNG image
                    generated_file_name=video_to_folder_full_path+"/frame_"+str(frame_count)+".png"
                    cv2.imwrite(generated_file_name, frame)
                    frame_count += 1

                    # we are saving the paths and respective labels seaprately here.
                    image_paths.append(generated_file_name)
                    image_labels.append(label)

                # Release the video capture object
                video.release()

#%%

# making .csv file with file_paths and their labels.
import pandas as pd

# Convert list to dictionary
data_dict = {'image_path':image_paths  , 'label':image_labels}

# Convert dictionary to pandas DataFrame
df = pd.DataFrame.from_dict(data_dict)

# Save DataFrame to CSV file
csv_file_path = 'csv_files/data.csv'
df.to_csv(csv_file_path)
print("CSV file has been created successfully.")
#%%
print(df)
column_stats = df.describe()
print("Column Statistics:")
print(column_stats)

#%%
# Print the count of each unique element in column 'A'
print("Count of each element in column 'A':")
print(df['label'].value_counts())