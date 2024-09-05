from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 1. 데이터
datasets = load_diabetes()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state1)


# 2. 모델 구성
model2 = RandomForestRegressor(random_state=random_state1)
model4 = XGBRegressor(random_state=random_state1)

models1 = [ model2]

for model in models1:
    model.fit(x_train, y_train)
    feature_importances = model.feature_importances_
    
    # 중요도 기반 정렬 (오름차순)
    sorted_idx = np.argsort(feature_importances)
    print(f"\n================= {model4.__class__.__name__} =================")
    print('Original R2 Score:', r2_score(y_test, model.predict(x_test)))
    
    # 하위 10%, 20%, 30%, 40%, 50% 제거하고 R2 스코어 계산
    for percentage in [10, 20, 30, 40, 50]:
        n_remove = int(len(sorted_idx) * (percentage / 100))
        removed_features_idx = sorted_idx[:n_remove]  # 하위 n% 특성 제거
        features_idx = sorted_idx[n_remove:-1]  # 하위 n% 특성 제거
        
        # 제거된 특성을 제외한 데이터셋 생성
        x_train_reduced = np.delete(x_train, removed_features_idx, axis=1)
        x_test_reduced = np.delete(x_test, removed_features_idx, axis=1)


        deleted_x_train = np.delete(x_train, features_idx, axis=1)
        deleted_x_test = np.delete(x_test, features_idx, axis=1)
        
        # PCA 적용
        n_components = 1  # 현재 남은 특성 개수
        pca = PCA(n_components=n_components)
        x_train_pca = pca.fit_transform(deleted_x_train)
        x_test_pca = pca.transform(deleted_x_test)
        
        # PCA 후 데이터를 결합
        x_train_final = np.concatenate([x_train_pca, x_train_reduced], axis=1)
        x_test_final = np.concatenate([x_test_pca, x_test_reduced], axis=1)
        
        # 모델 재학습 및 평가
        model4.fit(x_train_final, y_train)
        r2_pca = r2_score(y_test, model4.predict(x_test_final))
        
        print(f"R2 Score after removing {percentage}% lowest importance features and applying PCA: {r2_pca}")

'''
================= XGBRegressor =================
Original R2 Score: 0.3687286985683689
R2 Score after removing 10% lowest importance features and applying PCA: 0.20531206594164297
R2 Score after removing 20% lowest importance features and applying PCA: 0.16541376440595157
R2 Score after removing 30% lowest importance features and applying PCA: 0.1942695791636282
R2 Score after removing 40% lowest importance features and applying PCA: 0.09043281694271044
R2 Score after removing 50% lowest importance features and applying PCA: 0.13375102519541315
'''