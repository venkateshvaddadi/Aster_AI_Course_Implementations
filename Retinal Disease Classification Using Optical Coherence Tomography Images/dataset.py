import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np


# Custom Dataset class for classification
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, usecols=['image_path', 'label'])
        self.transform = transform

        # Mapping class names to integers
        self.class_to_int = {"NORMAL": 0, "DRUSEN": 1, "DME": 2,"CNV": 3}



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self.data['image_path'][idx]
        label = self.class_to_int[self.data['label'][idx]]  # Convert class label to integer

        # Load image using PIL
        image = Image.open(image_path)
        #image=np.array(image)
        #print(np.array(image).shape)
        if self.transform:
             image = self.transform(image)
        # #print(image.shape,label)

        # image=image[:3,:,:]

        #print(np.unique(image))
        return image, label
#%%
# import os
# print(os.listdir('./OCT_data/train/NORMAL'))
# print(os.listdir('./OCT_data/train/DRUSEN'))
# print(os.listdir('./OCT_data/train/DME'))
# print(os.listdir('./OCT_data/train/CNV'))

# # #%%
# # import torchvision.transforms as transforms

# # Define transformations for data augmentation or normalization
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize image to required size
#     transforms.ToTensor()
# ])
# # #%%


# # test_dataset = CustomDataset('./csv_files/data_test.csv', transform=transform)
# # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # print(len(test_dataloader))
# # #%%
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(device)

# # #%%
# # for batch_idx, (images, labels) in enumerate(test_dataloader):
# #     images = images.to(device)
# #     labels = labels.to(device)
# #     print(images.shape)
# #%%

# #%%


# from PIL import Image
# import numpy as np

# import pandas as pd
# data = pd.read_csv('./csv_files/data_train.csv', usecols=['image_path', 'label'])


# #%%
# # Mapping class names to integers
# class_to_int = {"NORMAL": 0, "DRUSEN": 1, "DME": 2,"CNV": 3}

# print("#"*100)


# for idx in range(len(data)):
#     row = data.iloc[idx]
#     image_path = data['image_path'][idx]
    
#     # Convert class label to integer
#     label = class_to_int[data['label'][idx]]
#     # Load image using PIL
#     image = Image.open(image_path)
#     # image=np.array(image)
#     image=transform(image)
#     if(len(image.shape)==3 ):
#         print(len(image.shape),image.shape,image_path)


#%%
from torchsummary import summary

summary(model, (3, 224, 224))






