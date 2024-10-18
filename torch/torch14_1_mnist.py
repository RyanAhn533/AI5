import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, dataloader
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE', DEVICE)

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor()])

#1. 데이터
path = './study/torch/_data'
#train_dataset = MNIST(path, train=True, download=False)
#test_dataset = MNIST(path, train=True, download=False)

train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)


print(train_dataset[0][0].shape) #<PIL.Imahe
print(train_dataset[0][1])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

###################### 잘 받아졌는지 확인 #######
bbb = iter(train_dataset)

aaa = next(bbb)
print(aaa)
print(aaa[0].shape)
print(len(train_loader))


#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            # model.COnv2D(64, (3,3), stirde=1, input_shape=(56,56,1))
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # (n, 64, 27, 27)
            nn.Dropout(0.5,)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=(3,3), stride=1), #(n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),  #(n, 32, 12, 12)
            nn.Dropout(0.5,)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32,16, kernel_size=(3,3), stride=1), #(n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5,)
        )
        
        self.hidden_layer3 = nn.Linear(16*5*5, 16)       
        self.output_layer = nn.Linear(in_features=32, out_features=10)
        
#    def forward(self, x):
#        x = 