

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\' 

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)



x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']]


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    random_state=100)



#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=8))
model.add(Dense(95, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(45, activation='rel'))
model.add(Dense(25, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)
y_predict = model.predict(x_test)

y_submit = model.predict(test_csv)
casual_predict = y_submit[:,0]
registered_predict = y_submit[:,1]

test_csv = test_csv.assign(casual = casual_predict, registered = registered_predict)
test_csv.to_csv(path + "test_bike2.csv")

print('로스 : ', loss)