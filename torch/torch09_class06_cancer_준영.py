import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

##########################랜덤 고정하자꾸나 ####################
SEED = 1004

import random
random.seed(SEED)   #파이썬 랜덤 고정
np.random.seed(SEED)   #넘파이 랜덤 고정

##토치 시드 고정
torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
#(569, 30) (569,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True,
                                                    random_state=369, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#############토치로 변환###################
x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

from torch.utils.data import TensorDataset #x, y 합친다
from torch.utils.data import dataloader #batch 정의 .

#토치 데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train) #합쳐지니까 tensor tuple형태
test_set = TensorDataset(x_test, y_test)
print(train_set)
print(len(train_set)) #398 = x_train과 y_train 행의 개수
print(train_set[0])
print(train_set[0][0]) # 첫번째 x
print(train_set[0][1]) # 첫번째 y train_set[397]까지 있겠지

#토치 데이터셋 만들기 2. batch 넣어준다. 끝

train_loader = dataloader(train_set, batch_size=40, shuffle=True)
test_loader = dataloader(train_set, batch_size=40, shuffle=False)
print(len(train_loader)) #batch 40개로 자르다보니 10개
print(train_loader)#<torch.utils.data.dataloader.DataLoader object at 0x00000236EC4C0A70>


#1. 이터레이터를 for문으로 확인
# for aaa in train_loader:
#     print(aaa)
#     break

# 지금까지 x y 합치고 batch 정의해주고 iterator의 정의를 만들었다.

bbb = iter(train_loader)
#aaa = bbb.next()  # 파이썬 3.9까지 먹히는 문법
aaa = next(bbb)
print(aaa)



# print("========================")
# print(x_train.shape, x_test.shape) #torch.Size([398, 30]) torch.Size([171, 1, 30])
# print(y_train.shape, y_test.shape) #torch.Size([398]) torch.Size([171, 1])
# print(type(x_train), type(y_train)) #<class 'torch.Tensor'> <class 'torch.Tensor'> 


# class Model(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         #super().__init__() #명시해줘도 되고 안해줘도 되낟 이건 디폴트다
#         super(Model, self).__init__()  #당분간은 그냥 하나의 덩어리로 써라
#         self.linear1 = nn.Linear(input_dim,64)
#         self.linear2 = nn.Linear(64,32)
#         self.linear3 = nn.Linear(32,16)
#         self.linear4 = nn.Linear(16,16)
#         self.linear5 = nn.Linear(16,output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.sigmoid = nn.Sigmoid()
#         #순전파
#     def forward(self, input_size):
#         x = self.linear1(input_size)
#         x = self.relu(x)    
    
#         x = self.linear2(x)
#         x = self.relu(x)    
#         x = self.linear3(x)
#         x = self.relu(x)    
#         x = self.linear4(x)
#         x = self.linear5(x)
#         x = self.sigmoid(x)
#         return(x)

# model = Model(30, 1).to(DEVICE)

# #3. 컴파일, 훈련
# criterion = nn.BCELoss()

# optimizer = optim.Adam(model.parameters(), lr=0.01)

# def train(model, criterion, optimizer, x, y):
#     optimizer.zero_grad()
#     hypothesis = model(x)
#     loss = criterion(hypothesis, y)
    
#     loss.backward()  #역전파 시작
#     optimizer.step()
#     return loss.item()

# epochs = 200
# for epoch in range(1, epochs + 1):
#     loss = train(model, criterion, optimizer, x_train, y_train)
#     print('epoch: {}, loss: {}'.format(epoch, loss))
    
# #4. 평가 예측
# #loss = model.evaluate(x, y)
# def evaluate(model, criterion, x, y):
#     model.eval()  #평가모드 // 역전파, 가중치 갱신, 기울기 계산 안하고싶지만 할수도있기도 없기도
#                   #drop out / batch normalization
#     with torch.no_grad():
#         y_predict = model(x)
#         loss2 = criterion(y_predict, y)
#     return loss2.item()

# last_loss = evaluate(model, criterion, x_test, y_test)
# print("최종 loss : ", last_loss)

# ##############################요 밑에 완성할 것 ####################
# from sklearn.metrics import accuracy_score
# result = model(x_test)
# acc = accuracy_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
# print('acc는?', acc)

# '''
# 최종 loss :  0.5850924849510193
# acc는? 0.9941520467836257
# '''