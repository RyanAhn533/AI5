from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping


path = "./_data/따릉이/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

print(x.shape)
print(y.shape)




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = StandardScaler() 
#MaxAbsScaler 로스는 ? 2150.105224609375
#r2스코어는?  0.7107245970696593
#RobustScaler 로스는 ? 2338.861572265625
#r2스코어는?  0.6853293162429761
 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

"""
#2-1 모델(순차형)
model = Sequential()
model.add(Dense(32, input_dim=9, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
"""
#2-2 모델 (함수형)
input1  = Input(shape=(9,))
dense1 = Dense(32, activation='relu', name='ys1')(input1)
dense2 = Dense(32, activation='relu', name='ys2')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(16, activation='relu', name='ys3')(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(16, activation='relu', name='ys4')(drop2)
dense5 = Dense(16, activation='relu', name='ys5')(dense4)
dense6 = Dense(8, activation='relu', name='ys6')(dense5)
dense7 = Dense(8, activation='relu', name='ys7')(dense6)
dense8 = Dense(4, activation='relu', name='ys8')(dense7)
output1 = Dense(1, name='ys9')(dense8)

model = Model(inputs=input1, outputs=output1)

#컴파일 훈련
model.compile(
    loss='mse',
    optimizer='adam')
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20,
                  restore_best_weights=True)
import time
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras32/keras32_Dropout4.h1"))

model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])
#model.save("./_save/keras30/keras30_4")
end = time.time()
#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print("걸린시간은?", "gpu on" if (len(gpus) > 0) else "gpu off", round(end - start, 2), "초")
if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다!")





"""
r2스코어는?  0.6087441395903814
걸린시간은? gpu on 4.05 초
쥐피유 돈다!!!

r2스코어는?  0.6657847507265329
걸린시간은? gpu off 3.09 초
쥐피유 없다!
"""