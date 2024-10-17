#Sources
#https://github.com/apache/spark/blob/master/examples/src/main/python/ml/tf_idf_example.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer
from pyspark.ml.feature import MinHashLSH
from pyspark.sql.functions import col

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
idf = IDF(inputCol="vectorized", outputCol="tf-idf", minDocFreq = 10)
model_idf = idf.fit(vectorized_df)
weighted_df = model_idf.transform(vectorized_df)


# Instantiate minhashing model
mh = MinHashLSH()
mh.setInputCol("tf-idf")
mh.setOutputCol("hashes")
mh.setSeed(2503)

# Fit model on tf-idf vectors
model = mh.fit(weighted_df)
model.setInputCol("tf-idf")

# Create a test item

#Asked ChatGPT: "Imagine you are a participant in the subreddit r/selfimprovement.
#Write a prompt discussing morality as an important aspect for pursuing self 
#improvement. Put it in a python string, lower case and spread through multiple lines

fake_post = (
    "morality guides our actions and decisions, providing a foundation for growth. "
    "when we align our self-improvement goals with our moral values, "
    "we not only enhance ourselves but also positively impact those around us. "
    "this alignment fosters a sense of purpose and fulfillment. "
    "self-improvement should not just be about personal gain; it should consider "
    "the well-being of others. embracing morality in our journey ensures that our "
    "efforts contribute to a better society and inspire others to do the same."
)


test = spark.createDataFrame([("test01", fake_post)], ["id", "cleaned_text"])
tokenized_test = tokenizer.transform(test)
vectorized_test = model_cv.transform(tokenized_test)
weighted_test = model_idf.transform(vectorized_test)


# 
model.approxSimilarityJoin(weighted_df, weighted_test, 0.7, distCol="JaccardDistance") \
     .select(
         col("datasetA.id").alias("id_reddit"),
         col("datasetB.id").alias("id_test"),
         col("JaccardDistance")) \
     .show()