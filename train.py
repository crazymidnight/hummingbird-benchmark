from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/creditcard.csv')

data = data.drop(columns=['Time'])
X_data, y_data = data.drop(columns=['Class']), data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.1, random_state=42, shuffle=False
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, shuffle=False
)
path = Path('data')
X_train.to_csv(path / 'X_train.csv', index=False)
y_train.to_csv(path / 'y_train.csv', index=False)
X_val.to_csv(path / 'X_val.csv', index=False)
y_val.to_csv(path / 'y_val.csv', index=False)
X_test.to_csv(path / 'X_test.csv', index=False)
y_test.to_csv(path / 'y_test.csv', index=False)

num_round = 50
model = lgb.LGBMClassifier()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=5,
    eval_metric='binary_logloss',
)
joblib.dump(model, 'model.pkl')
