from pyspark.sql import SparkSession

spark = SparkSession \
        .builder \
        .appName("lab1") \
        .getOrCreate()

#Read CSV file 
df = spark.read.option("delimiter", ";").option("header", True).csv("../self_improvement_cleaned.csv")
df.show()