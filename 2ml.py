import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
df = pd.read_csv("/content/CAR DETAILS FROM CAR DEKHO.csv") 
# Feature engineering: extract brand from 'name' 
df['brand'] = df['name'].str.split(' ').str[0] 
df.drop('name', axis=1, inplace=True) 
# Log-transform target variable 
df['selling_price'] = np.log1p(df['selling_price']) 
# One-hot encode categorical variables 
df = pd.get_dummies(df, drop_first=True) 
# Split into features and target 
X = df.drop('selling_price', axis=1) 
y = df['selling_price'] 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Pipeline 
pipe = Pipeline([ 
('scaler', StandardScaler()), 
('model', LinearRegression()) 
]) 
pipe.fit(X_train, y_train) 
y_pred = pipe.predict(X_test) 
# Convert predictions back to original price scale 
y_test_exp = np.expm1(y_test) 
y_pred_exp = np.expm1(y_pred) 
# Metrics 
rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp)) 
mae = mean_absolute_error(y_test_exp, y_pred_exp) 
r2 = r2_score(y_test_exp, y_pred_exp) 
# Normalized Metrics (relative error between 0–1) 
mean_price = y_test.mean() 
nrmse = rmse / mean_price 
nmae = mae / mean_price 
print("\nNormalized Metrics (Relative Errors):") 
print(f"Normalized RMSE: {nrmse:.4f}") 
print(f"Normalized MAE: {nmae:.4f}") 