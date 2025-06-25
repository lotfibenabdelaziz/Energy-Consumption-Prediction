import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

from data_preprocessing import load_and_preprocess_data

# Load the data
df = load_and_preprocess_data()
df['log_energy'] = np.log1p(df['total_energy_consumption'])

X = df.drop(['total_energy_consumption', 'log_energy'], axis=1)
y = df['log_energy']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model from disk
model = joblib.load("best_model/model.joblib")  

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Log with MLflow
# mlflow.log_metric("r2", r2)
# mlflow.log_metric("rmse", rmse)
# mlflow.log_metric("mae", mae)