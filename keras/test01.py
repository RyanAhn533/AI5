from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

# 데이터 로드
path = "C:\\프로그램\\ai5\\_data\\kaggle\\jena\\"
a = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# 데이터의 앞부분 420407개 행만 선택
a = a.head(420407)

# 'T (degC)' 열을 제거한 데이터프레임 (입력 변수들)
a1 = a.drop(['T (degC)'], axis=1)

# 'T (degC)' 열만 선택 (목표 변수)
a2 = a['T (degC)']

# a1과 a2의 크기 조정
a1 = a1.head(420406)
a2 = a2.tail(420406)

# 인덱스 재설정
a1 = a1.reset_index(drop=True)
a2 = a2.reset_index(drop=True)

# 데이터 결합
a = pd.concat([a1, a2], axis=1)

# Test 데이터 준비 (마지막 144개의 행)
b = a.tail(144)
b1 = b.drop(['T (degC)'], axis=1)
x1 = a.drop(['T (degC)'], axis=1)
y1 = a['T (degC)']

# 시계열 데이터 생성
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

size = 144

x = split_x(x1.values, size)  # .values로 numpy array로 변환
y = split_x(y1.values, size)


# Train/Test 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)

# 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(size, x1.shape[1]), return_sequences=True))
model.add(LSTM(64)) 
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(144))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=5,
    restore_best_weights=True
)

# x_train, y_train을 사용하여 학습
model.fit(x_train, y_train[:, -1], epochs=10, batch_size=2048, callbacks=[es])  # y_train의 마지막 열만 사용

# 평가, 예측
results = model.evaluate(x_test, y_test[:, -1],)
print('loss : ', results)

# 예측을 위한 x_pred
x_pred = np.array([b1.values],).reshape(1, 144,13)  # b1을 numpy array로 변환
y_pred = model.predict(x_pred)
print(' 결과', y_pred)
print(y_pred.shape)

# 성능 평가
r2 = r2_score(y_test[:, -1], y_pred.flatten())
print("r2스코어 : ", r2)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(y_test[:, -1], y_pred.flatten())
print("rmse = ", rmse)
