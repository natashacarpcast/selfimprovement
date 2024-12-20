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

#Ensure correct parsing of data
df.show(10)

#Moral Foundation Dictionary 2.0 scores
scores = ['Care_Virtue', 'Care_Vice', 'Fairness_Virtue',
       'Fairness_Vice', 'Loyalty_Virtue', 'Loyalty_Vice', 'Authority_Virtue',
       'Authority_Vice', 'Sanctity_Virtue', 'Sanctity_Vice']

#Make sure they're read as floats
df_features = df.select(*(F.col(c).cast("float").alias(c) for c in scores)).dropna()
df_features = df_features.withColumn('features', F.array(*[F.col(c) for c in scores]))\
                                                    .select('features')

#Try k-means first with PCA dimensionality reduction

#Create dense vector format for PCA and k means
vectors = df_features.rdd.map(lambda row: Vectors.dense(row.features))
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

#Perform clustering with k means. First create function
def clustering_pca(k):
    pca_kmeans = KMeans(k=k, seed=2503)
    pca_model = pca_kmeans.fit(pca_features)
    pca_pred = pca_model.transform(pca_features)
    pca_evaluator = ClusteringEvaluator()
    silhouette = pca_evaluator.evaluate(pca_pred)
    print("Silhouette with squared euclidean distance = {} when using PCA and {} k clusters".format(str(silhouette), k))
    print("Predictions corresponding to PCA and {} k clusters".format(k))
    pca_pred.groupby('prediction') \
               .count() \
               .show()



#Run it for different k 
k_to_try = [2,3,4,5,6,7,8,9,10]  
for k in k_to_try:
    clustering_pca(k)

#Unpersist to free memory for following steps
features.unpersist()
pca_features.unpersist()
    
#Now try using SVD to see which one performs better
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

#Create function to perform k means clustering, now with SVD
def clustering_svd(k):
        svd_kmeans = KMeans(k=k, seed=2503)
        svd_model = svd_kmeans.fit(U_df)
        svd_predictions = svd_model.transform(U_df)
        svd_evaluator = ClusteringEvaluator()
        silhouette = svd_evaluator.evaluate(svd_predictions)
        print("Silhouette with squared euclidean distance = {} when using SVD and {} k clusters".format(str(silhouette), k))
        print("Predictions corresponding to SVD and {} k clusters".format(k))
        svd_predictions.groupby('prediction') \
               .count() \
               .show()

for k in k_to_try:
    clustering_svd(k)


