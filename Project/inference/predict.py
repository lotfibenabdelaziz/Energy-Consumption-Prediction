import joblib
import pandas as pd

model = joblib.load("best_model.joblib")

def predict(input_df: pd.DataFrame):
    preds = model.predict(input_df)
    return preds

def load_model_and_predict(input_df):
    model = joblib.load("best_model.joblib")
    return model.predict(input_df)