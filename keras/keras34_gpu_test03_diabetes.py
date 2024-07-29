from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import sklearn as sk
from sklearn.datasets import load_diabetes
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

dataset = load_diabetes()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = RobustScaler()
#MaxAbsScaler 2980 0.44
#RobustScaler r2스코어는?  0.41430952830512535 로스는 ? [3171.337158203125
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#모델2-2(함수형)
input1 = Input(shape=(10,))
dense1 = Dense(32, activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(1, name='ys5')(dense4)

model = Model(inputs=input1, outputs=output1)


#컴파일 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20,
                  restore_best_weights=True)
import time
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras32/keras32_dropout.h1"))

model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])
#model.save("./_save/keras30/keras30_3")
end = time.time()
#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print("걸린시간은?", "gpu on" if (len(gpus) > 0) else "gpu off",  round(end -start, 2), '초')
#스케일링 전 0.3892
#스케일링 후 0.40
"""
로스는 ? [3913.365478515625, 0.0]
r2스코어는?  0.2772698222463196
걸린시간은? gpu on 4.42 초

r2스코어는?  0.3317791679638712
걸린시간은? gpu off 2.63 초
"""