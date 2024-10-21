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

df = spark.read.csv('../cleaned_emfd_liwc_results.csv', header=True)

#extended Moral Foundation Dictionary scores
scores = ['Care_Score', 'Fairness_Score', 'Loyalty_Score',
       'Authority_Score', 'Sanctity_Score', 'Care_Sentiment',
       'Fairness_Sentiment', 'Loyalty_Sentiment', 'Authority_Sentiment',
       'Sanctity_Sentiment']

#Make sure they're read as floats
df_features = df.select(*(F.col(c).cast("float").alias(c) for c in scores)).dropna()
df_features = df_features.withColumn('features', F.array(*[F.col(c) for c in scores]))\
                                                    .select('features')

#Do SVD dimensionality reduction 
vectors_rdd = df_features.rdd.map(lambda row: row["features"])
standardizer_rdd = StandardScalerRDD()
model = standardizer_rdd.fit(vectors_rdd)
vectors_rdd = model.transform(vectors_rdd)
mat = RowMatrix(vectors_rdd)
svd = mat.computeSVD(2, computeU=True)
U = svd.U
s = svd.s
V = svd.V
U_df = U.rows.map(lambda row: Row(features=Vectors.dense(row.toArray()))) \
             .toDF()
U_df.persist()

#Run K Mean clustering with already known best value
#Result from k testing:
#Silhouette with squared euclidean distance = 0.6154420047865076 when using SVD and 2 k clusters
svd_kmeans = KMeans(k=2, seed=2503)
svd_model = svd_kmeans.fit(U_df)
svd_predictions = svd_model.transform(U_df)
svd_evaluator = ClusteringEvaluator()
silhouette = svd_evaluator.evaluate(svd_predictions)
#Confirm silhouette
print("Silhouette with squared euclidean distance = {} when using SVD and 2 k clusters".format(str(silhouette)))


#Insert predictions in original df 
df_features_with_id = df_features.withColumn("id_clst", F.monotonically_increasing_id())
svd_predictions_with_id = svd_predictions.withColumn("id_clst", F.monotonically_increasing_id())
merged_df = df_features_with_id.join(svd_predictions_with_id, on="id", how="inner")

# Generate summary statistics of the scores for each cluster
summary_df = merged_df.groupBy('prediction').agg(
    *[F.mean(c).alias(f'mean_{c}') for c in scores] +  
    [F.stddev(c).alias(f'stddev_{c}') for c in scores])

summary_df.show()
