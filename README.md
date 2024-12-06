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

### Lab 3 - Association Rules Mining

1- [Association Rules Mining notebook](lab3_association_rules/association.ipynb) : This notebook explores looking at associations between morality language scores and emotional language scores, using the FP Growth algorithm on a Spark environment. For the morality language scores, different dimensions of moral values are included: care, authority, loyalty, sanctity and fairness. For the emotional language scores: positive emotion, anxiety, sad and anger language are included. 

2- [Parquet files with cluster's assignations from lab 2](lab3_association_rules/data_and_predictions)

### Lab 4 - Graphs

1-  All of the notebooks named text-network* are pyspark notebooks in which I created different text networks with varying conditions. For example, some of them had higher thresholds of TF IDF of words to be included in the network. I also played around with one of them only including nouns and adjectives. Other than those details, notebooks are essentially the same. After realizing the inefficiency, I created a [script-textnet.py](lab4_graphs/script-textnet.py) with its corresponding [sbatch file](lab4_graphs/script-textnet.sbatch) to simply change details there and send the job easier to the cluster for creating new networks. 

2-  The [neighbors.ipynb](lab4_graphs/neighbors.ipynb) was a very naive and simple exploration of which were the strongest neighbors of certain morality words in some of the networks version. 

3-  All of the edges_network* and nodes_network* folders are simply the csv files produced by the text-network* notebooks including information on nodes and edges.

4-  The community_detection* folders are the different attempts of community detection using the different folder. The one included in the final conclusions and the best one so far is the [sixth one](lab4_graphs/community_detection6.ipynb). As the text-network* notebooks, these are also all almost identical in structure and code. 

5-  The [node2vec.ipynb](lab4_graphs/node2vec.ipynb) does node2vec training and clustering with nodes with attributes, as explained in report. 

6-  The [communitydetectionliwc](lab4_graphs/communitydetectionliwc) folder contains the [exploration notebook](lab4_graphs/communitydetectionliwc/exploration.ipynb) where the groups from the community detection 6 + the node2vec clusters where further analyzed in terms of their proportion of morality and emotional language.



