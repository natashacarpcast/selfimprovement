from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.feature import StandardScaler as StandardScalerRDD
from pyspark.mllib.linalg.distributed import RowMatrix
import pyspark.sql.functions as F

spark = SparkSession \
        .builder \
        .appName("dr_cluster") \
        .getOrCreate()

df = spark.read.csv('cleaned_emfd_liwc_results.csv', header=True)

#extended Moral Foundation Dictionary scores
scores = ['Care_Score', 'Fairness_Score', 'Loyalty_Score',
       'Authority_Score', 'Sanctity_Score', 'Care_Sentiment',
       'Fairness_Sentiment', 'Loyalty_Sentiment', 'Authority_Sentiment',
       'Sanctity_Sentiment']

