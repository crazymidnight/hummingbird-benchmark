from pathlib import Path
from time import time

import joblib
import lightgbm as lgb
import pandas as pd
import torch
from hummingbird.ml import convert


path = Path('data')
X_test = pd.read_csv(path / 'X_test.csv')
y_test = pd.read_csv(path / 'y_test.csv')

lgb_model = joblib.load('model.pkl')
begin = time()
pred = lgb_model.predict(X_test)
total = time() - begin
print('LightGBM time:', total, 's')

torch_model = convert(lgb_model, 'pytorch')

begin = time()
torch_pred = torch_model.predict(X_test.to_numpy())
total = time() - begin
print('PyTorch time:', total, 's')

print('Are predictions equal:', pred == torch_pred)

torch_model = torch.jit.trace(torch_model, example_inputs=torch.randn(1, 30))
begin = time()
torch_pred = torch_model.forward(torch.tensor(X_test.to_numpy()))
total = time() - begin
print('PyTorch time:', total, 's')
import pprint

pprint.pprint(torch_pred)
print('Are predictions equal:', pred == torch_pred.to_numpy())
