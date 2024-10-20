import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]).transpose()
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape, y.shape)
# (10, 2) (10,)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
print(x.shape, y.shape)
exit()
#2. 모델 정의
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.Linear(10, 9),
    nn.Linear(9, 8),
    nn.Linear(8, 7),
    nn.Linear(7, 6),
    nn.Linear(6, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
).to(DEVICE)

#3. 컴파일
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#4. 훈련 함수
def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

#5. 훈련 루프
epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

def evaluate(model, criterion, x, y):        
    model.eval()
        
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
        
#7. 최종 손실과 예측 출력
loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

#8. 새로운 값 예측
new_input = torch.Tensor([[4, 1.3]]).to(DEVICE)  # 두 개의 입력을 받도록 조정

print('4의 예측값 :', results.item())
