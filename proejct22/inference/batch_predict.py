import mlflow.sklearn
import pandas as pd

def load_model_and_predict(input_df):
    model = mlflow.sklearn.load_model('runs:/your_run_id/model')
    return model.predict(input_df)