import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor, CatBoostClassifier



#1. 데이터
x, y = fetch_covtype(return_X_y=True)
random_state = 777
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, train_size=0.8,
                                                    stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. Bayesian Optimization을 위한 파라미터 설정
bayesian_params = {
    'learning_rate': (0.001, 0.1),
    'depth': (4, 10),
    'l2_leaf_reg': (1, 10),
    'bagging_temperature': (0.0, 1.0),
    'subsample': (0.5, 1),
}

# 3. 블랙박스 함수 정의
def catboost_hamsu(learning_rate, depth, l2_leaf_reg, bagging_temperature, subsample):
    params = {
        'iterations': 100,
        'learning_rate': learning_rate,
        'depth': int(depth),
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temperature,
        'task_type': 'GPU',
        'devices': '0',
        'early_stopping_rounds': 5,
        'verbose': 0
    }
    
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

# 4. Bayesian Optimization 설정
bay = BayesianOptimization(
    f=catboost_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

# 5. 최적화 실행
n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

# 6. 최적 파라미터 및 실행 시간 출력
print(bay.max)
print(f"{n_iter}번 걸린 시간은? {round(end_time - start_time, 2)} 초")
'''
 105       | 0.4732    | 0.9047    | 4.624     | 9.887     | 0.08915   | 0.6224    |
=====================================================================================
{'target': 0.6754748522150904, 'params': {'bagging_temperature': 0.8426187158614642, 'depth': 10.0,
'l2_leaf_reg': 4.404174933008292, 'learning_rate': 0.1, 'subsample': 1.0}}
'''