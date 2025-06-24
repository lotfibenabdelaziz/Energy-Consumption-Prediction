import mlflow
from pyspark.sql.functions import struct, col
logged_model = 'runs:/6fad7ce1c0134d0b8ab8dca931773bd2/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))