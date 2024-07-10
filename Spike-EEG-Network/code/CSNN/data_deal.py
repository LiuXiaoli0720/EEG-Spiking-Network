import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
data_loader=np.load('/root/yxy/ws/DUL/dataset/0.5s/S1.npz', allow_pickle=True)

class TrainDataset(Dataset):
    def __init__(self,transforms):

        self.train_images=data_loader['train_images']
        self.train_labels=data_loader['train_label']
        # print(self.train_images.shape)
        # self.train_images=self.train_images.squeeze(-1)
        self.transform = transforms
    
    def __len__(self):
        return self.train_images.shape[0]
    
    def __getitem__(self,index):
        
        return self.transform(self.train_images[index][:,:64]),self.train_labels[index]
    
class TestDataset(Dataset):
    def __init__(self,transforms):
        self.test_images=data_loader['test_images']
        self.test_labels=data_loader['test_label']
        # self.test_images=self.test_images.squeeze(-1)
        self.transform = transforms
    
    def __len__(self):
        return self.test_images.shape[0]
    
    def __getitem__(self,index):
        return self.transform(self.test_images[index][:,:64]),self.test_labels[index]

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset=TrainDataset(transforms=transform_train)
test_dataset=TestDataset(transforms=transform_test)


print(train_dataset[0][0].max(),train_dataset[0][1])
print(train_dataset[0][0].shape)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True,pin_memory=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True,pin_memory=True)