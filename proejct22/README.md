"""
# Electricity Forecasting Project

This project forecasts energy demand using ML techniques.

## Structure
- `data_preprocessing.py`: Load and prepare features.
- `train.py`: Train ML models using GridSearchCV and log them to MLflow.
- `inference/`: Scripts for making predictions with the trained model.

## How to Run
1. Prepare your dataset.
2. Run `train.py` to train and log models.
3. Use `spark_predict.py` or `batch_predict.py` for inference.
"""