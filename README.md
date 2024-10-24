# selfimprovement

### Lab 1 - Similarity Measurements

Files should be viewed in this order:

1- [TF-IDF experimentation](tfidf.py): This file takes a dataframe of reddit posts on the r/selfimprovement. Converts it to TF-IDF vectors and then uses LSH for identifying posts that could be similar to a fake post that talks about morality. 

2- [Word2Vec experimentation](word2vec.py): This file takes a dataframe of reddit posts on the r/selfimprovement. Converts it to word2vec vectors and then uses LSH for identifying posts that could be similar to a fake post that talks about morality. 

3 - [TF-IDF batch script](tfidf.sbatch) and [word2vec batch script](word2vec.sbatch) to run jobs on midway cluster.

4 - [Output of TF-IDF experimentation](tfidf.out) and [output of word2vec experimentation](word2vec.out)

5 - [Notebook](exploring_reddit.ipynb) exploring the found similar reddit posts from TF-IDF LSH. 

### Lab 2 - Clustering and Dimensionality Reduction 

Inside the [lab2_clustering folder](lab2_clustering), files should be viewed in this order: 

1 - [mfd2_findk.py](lab2_clustering/mfd2_findk.py), [mfd2_findk.sbatch](lab2_clustering/mfd2_findk.sbatch) and [mfd2_findk.out](lab2_clustering/mfd2_findk.out): These are the python file, sbatch file and its corresponding output file. At the python file, I performed PCA and SVD for several runs of K-Means clustering, aiming to find the optimal combination for the model's better performance. 

2 - [mfd2_final.py](lab2_clustering/mfd2_final.py), [mfd2_final.sbatch](lab2_clustering/mfd2_final.sbatch) and [mfd2_final.out](lab2_clustering/mfd2_final.out): This python, sbatch and output file correspond to the following task -->  After identifying the best model, it was ran again on a different file in order to get [summary statistics](lab2_clustering/summary_mfd2_clusters.csv), visualize some examples of the documents in each cluster, and to create a [parquet file](lab2_clustering/data_and_predictions) containing the prediction for each document. 

3 - The [explore_clusters notebook](lab2_clustering/explore_clusters.ipynb) looks at the summary csv file again, and displays 5 example documents from each cluster

4 - The [top-n-words notebook](lab2_clustering/top-n-words.ipynb) looks at the top 100 words in each cluster. 


