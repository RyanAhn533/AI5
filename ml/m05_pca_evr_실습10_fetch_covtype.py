from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import sklearn as sk
from sklearn.datasets import fetch_covtype
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
###############


datasets = fetch_covtype()
x = datasets.data
y = datasets.target


print(y)
print(y.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x)

x = x.reshape(581012,54)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=3)


x = np.concatenate([x_train, x_test], axis=0)
from sklearn.decomposition import PCA
pca = PCA(n_components=54)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print('0.95이상 : ', np.argmax(evr_cumsum>=0.95)+1)
print('0.99이상 : ', np.argmax(evr_cumsum>=0.99)+1)
print('0.999이상 : ', np.argmax(evr_cumsum>=0.999)+1)
print('0.1이상 : ', np.argmax(evr_cumsum>=1)+1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler


for i in range(4) : 
    x = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
    pca = PCA(n_components= x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    print(x_train1.shape, x_test1.shape)
    
    
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
#scaler = StandardScaler()

#scaler = MaxAbsScaler() 0.61 r2 0.49
#scaler = RobustScaler() r2 0.304  loss 0.91951
#모델
    model = Sequential()
    model.add (Dense(512, input_shape=(x[i],)))
    model.add (Dense(256, activation='relu'))
    model.add (Dense(256, activation='relu'))
    model.add (Dense(1))



    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=0, 
                   restore_best_weights=True)

    ##### mcp 세이브 파일명 만들기 #####
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = 'C:\\프로그램\\ai5\\_save\\ml04\\'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml04_', str(i+1), '_', date, '_', filename])  

    mcp = ModelCheckpoint(
            monitor='val_loss',
            mode='auto',
            verbose=0,     
            save_best_only=True,   
            filepath=filepath,)
    import time

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=200, batch_size=2048,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    y_predict = model.predict(x_test1)
    print(y_test.shape, y_predict.shape)
    
    
    
    
    r2 = r2_score(y_test, y_predict)
    print("===============================================")
    print('결과', i+1)
    print('PCA :',x[i])
    print('r2 :', r2)
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


'''
===============================================
결과 1
PCA : 43
r2 : 0.8070150934326423
걸린 시간 : 35.31 초
===============================================
(406708, 49) (174304, 49)
(174304,) (174304, 1)
===============================================
결과 2
PCA : 49
r2 : 0.564186276752199
걸린 시간 : 4.89 초
===============================================
(406708, 51) (174304, 51)
(174304,) (174304, 1)
===============================================
결과 3
PCA : 51
r2 : 0.7924045179300103
걸린 시간 : 11.28 초
===============================================
(406708, 52) (174304, 52)
(174304,) (174304, 1)
===============================================
결과 4
PCA : 52
r2 : 0.8119654013548732
걸린 시간 : 11.51 초
===============================================
'''