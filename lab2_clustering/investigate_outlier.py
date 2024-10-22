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

df = spark.read.csv('../cleaned_mfd2_liwc_results.csv', header=True)

print(df.columns)

#extended Moral Foundation Dictionary scores
scores = ['Care_Virtue', 'Care_Vice', 'Fairness_Virtue',
       'Fairness_Vice', 'Loyalty_Virtue', 'Loyalty_Vice', 'Authority_Virtue',
       'Authority_Vice', 'Sanctity_Virtue', 'Sanctity_Vice']

#Make sure they're read as floats
df_features_og = df.select(*(F.col(c).cast("float").alias(c) for c in scores), 'id').dropna()
df_features_og = df_features_og.withColumn('features', F.array(*[F.col(c) for c in scores]))\
                                                    .select('id','features')

#Create dense vector format for PCA and k means
vectors = df_features_og.rdd.map(lambda row: Vectors.dense(row.features))
features = spark.createDataFrame(vectors.map(Row), ["features_unscaled"])

#Standardize as the main scores and the sentiments have different scales
standardizer = StandardScaler(inputCol="features_unscaled", outputCol="features")
model = standardizer.fit(features)
features = model.transform(features) \
                .select('features')

#Persist in memory for faster execution
features.persist()

#Perform dimensionality reduction
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features)
pca_results = model.transform(features).select("pcaFeatures")
pca_features = pca_results.rdd.map(lambda row: Vectors.dense(row.pcaFeatures))
pca_features = spark.createDataFrame(pca_features.map(Row), ["features"])
pca_features.persist()

#Run KMeans clustering
pca_kmeans = KMeans(k=3, seed=2503)
pca_model = pca_kmeans.fit(pca_features)
pca_pred = pca_model.transform(pca_features)
pca_evaluator = ClusteringEvaluator()
silhouette = pca_evaluator.evaluate(pca_pred)
print("Silhouette with squared euclidean distance = {} when using PCA and {} k clusters".format(str(silhouette), 3))
print("Predictions corresponding to PCA and {} k clusters".format(3))
pca_pred.groupby('prediction') \
               .count() \
               .show()

#Unpersist to free memory for following steps
features.unpersist()
pca_features.unpersist()

#Insert predictions in original df 
df_features_with_id = df_features_og.withColumn("id_clst", F.monotonically_increasing_id())
pca_predictions_with_id = pca_pred.withColumn("id_clst", F.monotonically_increasing_id())
merged_df = df_features_with_id.join(pca_predictions_with_id, on="id_clst", how="inner")
merged_df.show(10)

# Generate summary statistics of the scores for each cluster
summary_df = merged_df.groupBy('prediction').agg(
   *[F.mean(c).alias(f'mean_{c}') for c in scores] +  
   [F.stddev(c).alias(f'stddev_{c}') for c in scores])

summary_df.show()

#Visualize some items in each cluster
merged_df.filter(F.col('prediction') == 0) \
                    .show(10)

merged_df.filter(F.col('prediction') == 1) \
                    .show(10)

merged_df.filter(F.col('prediction') == 2) \
                    .show(10)


