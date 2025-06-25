import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Load the preprocessed dataset
from data_preprocessing import load_and_preprocess_data
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("EXPERIMENT_NAME", "default"))


def find_best_model_using_gridsearchcv(X, Y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=10)

    models_and_parameters = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'positive': [False]  # Avoid True as it adds constraint overhead
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 1, 10]
            }
        },
        'Lasso': {
            'model': Lasso(max_iter=1000),
            'params': {
                'alpha': [0.01, 0.1, 1],
                'selection': ['cyclic']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'max_depth': [5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [100],
                'max_features': [3],
                'max_depth': [10],
                'min_samples_split': [2],
                'min_samples_leaf': [1]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(eval_metric='rmse', verbosity=0, use_label_encoder=False),
            'params': {
                'n_estimators': [100],
                'max_depth': [3, 4],
                'learning_rate': [0.05, 0.1]
            }
        }
    }

    results = []

    for name, config in models_and_parameters.items():
        print(f"Training {name}...")
        gs = GridSearchCV(config['model'], config['params'], cv=cv, n_jobs=-1)
        gs.fit(X, Y)

        # Log with MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params(gs.best_params_)
            mlflow.log_metric("best_score", gs.best_score_)
            mlflow.sklearn.log_model(gs.best_estimator_, "model")

        results.append({
            'model': name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(results, columns=['model', 'best_score', 'best_params'])

if __name__ == '__main__':
    df = load_and_preprocess_data()

    # Target and Features
    X = df.drop(['total_energy_consumption', 'log_energy'], axis=1)
    Y = df['log_energy']  # log-transformed target

    best_models_df = find_best_model_using_gridsearchcv(X, Y)
    print(best_models_df)
    best_models_df.to_csv("best_models_results.csv", index=False)
