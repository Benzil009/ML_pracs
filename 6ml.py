import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
# Load dataset
df = pd.read_csv("final_dataset.csv")
 Define features and target
X = df.drop(columns=['AQI'])
y = df['AQI']
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Bagging with Decision Trees
bagging_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=42
)
# Train
bagging_model.fit(X_train, y_train)
# Predict
y_pred_bag = bagging_model.predict(X_test)
# Evaluate
r2_bag = r2_score(y_test, y_pred_bag)
mae_bag = mean_absolute_error(y_test, y_pred_bag)
rmse_bag = np.sqrt(mean_squared_error(y_test, y_pred_bag))
print("🌳 BAGGING RESULTS:")
print(f"R²: {r2_bag:.3f}")
print(f"MAE: {mae_bag:.2f}")
print(f"RMSE: {rmse_bag:.2f}")

Output : 🌳 BAGGING RESULTS:
R²: 0.935
MAE: 18.84
RMSE: 28.97
# Gradient Boosting
boost_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
# Train
boost_model.fit(X_train, y_train)

# Predict
y_pred_boost = boost_model.predict(X_test)

# Evaluate
r2_boost = r2_score(y_test, y_pred_boost)
mae_boost = mean_absolute_error(y_test, y_pred_boost)
rmse_boost = np.sqrt(mean_squared_error(y_test, y_pred_boost))

print("\n⚡ BOOSTING RESULTS:")
print(f"R²: {r2_boost:.3f}")
print(f"MAE: {mae_boost:.2f}")
print(f"RMSE: {rmse_boost:.2f}")

Output:
⚡ BOOSTING RESULTS:
R²: 0.941
MAE: 18.64
RMSE: 27.59
results = pd.DataFrame({
    'Model': ['Bagging', 'Boosting'],
    'R² Score': [r2_bag, r2_boost],
    'MAE': [mae_bag, mae_boost],
    'RMSE': [rmse_bag, rmse_boost]
})

print("\n📊 Accuracy Comparison:")
print(results)
