from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col
import mlflow

def main():
    # Step 1: Create Spark session
    spark = SparkSession.builder.appName("MLflowSparkInference").getOrCreate()

    # Step 2: Load your MLflow model URI from the run you want to use
    logged_model = 'runs:/f77be8131ae04ad6a0161122541f0e55/model'

    # Step 3: Load model as Spark UDF
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

    # Step 4: Load or create your Spark DataFrame (replace with your real data source)
    # Example: create dummy dataframe with columns matching your model input features
    data = [
        (1.2, 3.4, 5.6),
        (2.3, 4.5, 6.7)
    ]
    columns = ['feature1', 'feature2', 'feature3']  # replace with your real feature names
    df = spark.createDataFrame(data, schema=columns)

    # Step 5: Use the loaded model UDF to add predictions column
    df = df.withColumn('predictions', loaded_model(struct(*[col(c) for c in columns])))

    # Step 6: Show predictions
    df.show()

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()

# Run the script
# python spark_predict.py