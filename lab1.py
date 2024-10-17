#Sources
#https://github.com/apache/spark/blob/master/examples/src/main/python/ml/tf_idf_example.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer
from pyspark.ml.feature import MinHashLSH

spark = SparkSession \
        .builder \
        .appName("lab1") \
        .getOrCreate()

#Read CSV file 
df = spark.read.option("header", True).csv("just_text_submissions.csv")

#Tokenize
tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokenized")
tokenized_df = tokenizer.transform(df)

#Vectorize
cv = CountVectorizer(inputCol="tokenized", outputCol="vectorized")
model_cv = cv.fit(tokenized_df)
vectorized_df = model_cv.transform(tokenized_df)

#TF-IDF vectors
idf = IDF(inputCol="vectorized", outputCol="tf-idf")
model_idf = idf.fit(vectorized_df)
weighted_df = model_idf.transform(vectorized_df)
weighted_df.show()