import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

#1. 데이터
path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음


x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

print(x.shape, y.shape)  #(652, 8) (652,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True,
                                                    random_state=369, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
#x_train = torch.DoubleTensor(x_train).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
#y_train = torch.IntTensor(y_train).unsqueeze(1).to(DEVICE)
#y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)

x_test = torch.FloatTensor(x_test).to(DEVICE)
#x_test = torch.DoubleTensor(x_test).unsqueeze(1).to(DEVICE)

y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
#y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)
#y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)


#y_predict -> vector형태라서 reshape해줘야함

#int - long
#floattensor - doubletensor
#longtensor랑 floattensor언제 쓰는지 확인

print("========================")
print(x_train.shape, x_test.shape) #torch.Size([398, 30]) torch.Size([171, 1, 30])
print(y_train.shape, y_test.shape) #torch.Size([398]) torch.Size([171, 1])
print(type(x_train), type(y_train)) #<class 'torch.Tensor'> <class 'torch.Tensor'> 

#2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()    
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()  #역전파 시작
    optimizer.step()
    return loss.item()

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
#4. 평가 예측
#loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()  #평가모드 // 역전파, 가중치 갱신, 기울기 계산 안하고싶지만 할수도있기도 없기도
                  #drop out / batch normalization
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y_predict, y)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", last_loss)

##############################요 밑에 완성할 것 ####################
from sklearn.metrics import accuracy_score
result = model(x_test) #y_pred
#result = np.round(model(x_test)) #y_pred

acc = accuracy_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
#print('acc는?', acc)

y_test = y_test.cpu().numpy()

#acc = accuracy_score(y_test, result)
#print('acc : ?', acc)


print('acc는?', acc)

#최종 loss :  4.650130271911621
#acc는? 0.6785714285714286