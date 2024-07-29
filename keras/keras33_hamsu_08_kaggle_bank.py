from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler

# 파일 경로 설정
path = "./_data/kaggle/Bank/"

# CSV 파일 로드
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

# 데이터 확인
print(train_csv['Geography'].value_counts())

# 데이터 변환
train_csv['Geography'] = train_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
test_csv['Geography'] = test_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
train_csv['Gender'] = train_csv['Gender'].replace({'Male': 1, 'Female': 2})
test_csv['Gender'] = test_csv['Gender'].replace({'Male': 1, 'Female': 2})

# 특정 열에 0 값을 가진 행 삭제
"""
train_csv = train_csv[train_csv['Balance'] != 0]
test_csv = test_csv[test_csv['Balance'] != 0]
"""

# 문자열 값을 가진 열 확인 및 삭제
print(train_csv.select_dtypes(include=['object']).columns)
print(test_csv.select_dtypes(include=['object']).columns)

# 'Surname' 열 삭제
train_csv = train_csv.drop(['Surname'], axis=1)
test_csv = test_csv.drop(['Surname'], axis=1)

# 데이터 저장
train_csv.to_csv(path + "replaced_train.csv")
test_csv.to_csv(path + "replaced_test.csv")

# 데이터 로드
re_train_csv = pd.read_csv(path + "replaced_train.csv", index_col=0)
re_test_csv = pd.read_csv(path + "replaced_test.csv", index_col=0)

# 데이터 확인
re_train_csv.info()
re_test_csv.info()

# 특정 열 제거
re_train_csv = re_train_csv.drop(['CustomerId'], axis=1)
re_test_csv = re_test_csv.drop(['CustomerId'], axis=1)


# 데이터 스케일링
scaler = StandardScaler()
re_train_csv_scaled = scaler.fit_transform(re_train_csv.drop(['Exited'], axis=1))
re_test_csv_scaled = scaler.transform(re_test_csv)

"""
MaxAbsScaler
로스값은 :  0.32643696665763855
정확도는 :  0.8613021480292059
R2 스코어 : 0.16241584680842946

RobustScaler
로스값은 :  0.3278585970401764
정확도는 :  0.8619080801042203
R2 스코어 : 0.16607501742088715
"""

# 데이터프레임으로 변환
re_train_csv = pd.concat([pd.DataFrame(re_train_csv_scaled), re_train_csv['Exited'].reset_index(drop=True)], axis=1)
re_test_csv = pd.DataFrame(re_test_csv_scaled)



# 학습 데이터 분리
x = re_train_csv.drop(['Exited'], axis=1)
y = re_train_csv['Exited']
print(type(x))  # <class 'pandas.core.frame.DataFrame'>
print(type(y))  # <class 'pandas.core.series.Series'>

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1186, train_size=0.8)
"""
# 모델 구성
model = Sequential([
    Dense(16, input_dim=x_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
"""

#모델구성 2-2(함수형)
input1 = Input(shape=(10,))
dense1 = Dense(10, activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(1, name='ys5')(dense4)

model = Model(inputs=input1, outputs=output1)

# 컴파일 및 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=False)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras32/keras32.dropout8.h1"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start = t.time()
model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=1, validation_split=0.3, callbacks=[es])
end = t.time()


# 평가 및 예측
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test)
y_pred_rounded = np.round(y_pred)

# R2 및 정확도 계산
r2 = r2_score(y_test, y_pred_rounded)
accuracy = accuracy_score(y_test, y_pred_rounded)

# 예측 결과
y_submit = model.predict(re_test_csv)
y_submit = np.round(y_submit)

print('로스값은 : ', loss)
print('정확도는 : ', accuracy)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end-start, 2), '초')

# 제출 파일 생성
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
sample_submission_csv['Exited'] = y_submit
sample_submission_csv.to_csv(path + "submission_0723_3.csv")

"""
로스값은 :  0.328587144613266
정확도는 :  0.8608476989729451
R2 스코어 : 0.15967146884908612
걸린 시간 : 4.21 초

로스값은 :  0.3339275121688843
정확도는 :  0.8612112582179537
R2 스코어 : 0.1618669712165608
걸린 시간 : 8.65 초


로스값은 :  0.3488925099372864
정확도는 :  0.8564849880328416
R2 스코어 : 0.13332544043939065
걸린 시간 : 7.11 초
"""