from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
import pandas as pd

spark=SparkSession.builder\
.appName('selfimprovement').getOrCreate()


df = pd.read_csv('cleaned_csv/self_improvement_cleaned.csv')
df = spark.createDataFrame(df)

cv = CountVectorizer(inputCol='tokenized',
                     outputCol='features',
                     vocabSize=500, 
                     minDF= 0.05,
                     maxDF= 0.90)

cv_model = cv.fit(df)
vectorized_tokens = cv_model.transform(df)

lda = LDA(k=10)
model = lda.fit(vectorized_tokens)

vocab = cv_model.vocabulary
topics = model.describeTopics()

topics.rdd\
.map(lambda row: row['termIndices'])\
.collect()

topic_words = topics.rdd\
.map(lambda row:row['termIndices'])\
.map(lambda idx_list: [vocab[idx] for idx in idx_list])\
.collect

# Print the topic words
for idx, topic in enumerate(topic_words):
    print(f"Topic {idx}: {topic}")
