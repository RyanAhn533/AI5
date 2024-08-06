import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
import tensorflow as tf

train_datagen1 = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/'


xy_train1 = train_datagen1.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='binary',

color_mode='rgb',
shuffle=True).next()



train_datagen2 = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/cat_and_dog/train/'


xy_train2 = train_datagen2.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='binary',

color_mode='rgb',
shuffle=True).next()


xy_train = np.concatenate((xy_train1, xy_train2), axis = 0)

(x_train, y_train), (x_test, y_test) = xy_train()
####################################################################
x_train = x_train/255.
x_test = x_test/255.

test_datagen1 = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/'


xy_test1 = test_datagen1.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='binary',

color_mode='rgb',
shuffle=True).next()



test_datagen2 = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/cat_and_dog/train/'


xy_test2 = test_datagen2.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='binary',

color_mode='rgb',
shuffle=True).next()


xy_test = np.concatenate((xy_test1, xy_test2), axis = 0)

(x_train, y_train), (x_test, y_test) = xy_train()

x_train = x_train/255.
x_test = x_test/255.


train_datagen =  ImageDataGenerator(
    #rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)
np_path = 'C:/프로그램/ai5/_data/mixed_data/'
np.save(np_path + 'cat_dog_train.npy', arr=xy_train)

np_path = 'C:/프로그램/ai5/_data/mixed_data/'
np.save(np_path + 'cat_dog_test.npy', arr=xy_test)


exit()
augment_size = 5000
randidx = np.random.randint(x_train.shape[0], size = augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
                                  x_augmented.shape[0], 
                                  x_augmented.shape[1], 
                                  x_augmented.shape[2], 3)
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,).next()[0]

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

x_train = np.concatenate((x_train,x_augmented), axis = 0)
print(x_train.shape)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
exit()
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 모델 로드
model_path = 'c:/프로그램/ai5/_save/keras42/_kaggle_cats_dog/k42_01_0804_1847_0025-0.5556.hdf5'
model = load_model(model_path)

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 평가 예측
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

# 예측 값 생성 및 반올림
y_pre = np.round(model.predict(x_test, batch_size=16))

end_time = time.time()
print("걸린 시간 :", round(end_time-start,2),'초')

# CSV 파일 만들기
sampleSubmission = pd.read_csv('C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/sample_submission.csv', index_col=0)

# 예측 값 처리
y_submit = model.predict(x_test, batch_size=16)

# sampleSubmission과 일치하는 길이로 자르기 (필요시)
y_submit = y_submit[:len(sampleSubmission)]

# 제출 파일 작성
sampleSubmission['label'] = y_submit
sampleSubmission.to_csv('C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/' + "teacher0805_2.csv")
