import mlflow
from pyspark.sql.functions import struct, col

def run_spark_prediction(spark, df):
    logged_model = 'runs:/your_run_id/model'
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
    return df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))