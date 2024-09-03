import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk

print('버전', sk.__version__)

tf.random.set_seed(33)


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(178,13), (178)

print(np.unique(y, return_counts=True))


# 데이터 조정
x = x[:-39]
y = y[:-39]
print(np.unique(y, return_counts=True))

# OneHotEncoder는 제거하고 to_categorical 사용
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123, stratify=y)
#항상 train 데이터만 SMOTE한다 과적합된다.
########## SMOTE 적용 ##########
# cmd > pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk
print(sk.__version__)   # 1.5.1

print('증폭전:', np.unique(y_train, return_counts=True))
# 증폭전: (array([0, 1, 2]), array([44, 53,  6], dtype=int64))

ROS = RandomOverSampler(random_state=7777)
x_train, y_train = ROS.fit_resample(x_train, y_train)
print('증폭후:', np.unique(y_train, return_counts=True))
# 증폭후: (array([0, 1, 2]), array([53, 53, 53], dtype=int64))


######################   SMOTE 적용 ######################
#근접한 애에다가 새로운걸 쪽쪽쪽쪽 만든다.
#2. 모델 설정 (Keras 모델 예시)
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# f1_score 계산을 위해 y_test와 y_predict를 다시 정수형으로 변환
y_predict = model.predict(x_test)
y_predict_classes = np.argmax(y_predict, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_classes, y_predict_classes)
f1 = f1_score(y_test_classes, y_predict_classes, average='macro')

print('acc : ', acc)
print('f1 : ', f1)

'''
스모팅 후
loss :  0.3675796389579773
acc :  0.8571428656578064
acc :  0.8571428571428571
f1 :  0.5993265993265994
'''