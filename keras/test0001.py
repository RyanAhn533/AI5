# https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv
# 1-3열 index처리
# 문자를 수치화 해주기 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time


path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음
print(train_csv.shape) 
print(test_csv.shape)

encoder = LabelEncoder()



test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])

x = train_csv.drop(['CustomerId','Surname','Exited'], axis=1)
y = train_csv['Exited']

# from sklearn.preprocessing import MinMaxScaler
# scalar=MinMaxScaler()
# x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle= True, random_state= 512)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# print(x_train.shape)
# print(x_test.shape)

# print(x_test)
# print(test_csv)


#2 모델구성
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=10))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3 컴파일 훈련
from sklearn.metrics import accuracy_score


model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
start_time = time.time()
es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 30,
    restore_best_weights= True    
)

model.fit(x_train, y_train, epochs= 1000, batch_size=100, verbose=1, validation_split= 0.25, callbacks=[es])
end_time = time.time()

#4 평가 예측,
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
print("로스 : ", loss[0])
print("acc : ", round(loss[1], 3))

y_pred = np.round(y_pred)

accuracy_score(y_test, y_pred)

print("acc스코어 : ", accuracy_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초" )

y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
y_submit_binary = np.round(y_submit).astype(int)

#5 파일 생성
sampleSubmission['Exited'] = y_submit_binary

sampleSubmission.to_csv(path+'samplesubmission_0723_1239.csv')

print(sampleSubmission['Exited'].value_counts())
