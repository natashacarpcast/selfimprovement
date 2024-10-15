from pyspark.sql import SparkSession

spark = SparkSession \
        .builder \
        .appName("lab1") \
        .getOrCreate()

#Read CSV file 
df = spark.read.csv("just_text_submissions.csv")
df.show()