
from pyspark.sql import SparkSession
from graphframes import *
import sparknlp
from sparknlp.base import DocumentAssembler
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
import itertools
from pyspark.sql.functions import col, least, greatest, lit
from sparknlp.base import Finisher

spark = SparkSession \
        .builder \
        .appName("network") \
        .getOrCreate()

data = spark.read.csv("../data/cleaned_moral_scores.csv", header= True).select(["id", "cleaned_text"])

english = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", 
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "did", 
    "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", 
    "its", "itself", "let", "me", "more", "most", "must", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "some", "such", 
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", 
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves", "will", "ll", 
    "re", "ve", "d", "s", "m", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", 
    "s", "t", "u", "v", "w", "x", "y", "z", "many", "us", "ok", "hows", "ive", "ill", "im", "cant", "topics", "topic",
    "discuss", "thoughts", "yo", "thats", "whats", "lets", "nothing", "oh", "omg", 
         "things", "stuff", "yall", "haha", "yes", "no", "wo", "like", 'good', 
         'work', 'got', 'going', 'dont', 'really', 'want', 'make', 'think', 
         'know', 'feel', 'people', 'life', "getting", "lot" "great", "i", "me", 
         "my", "myself", "we", "our", "ours", "ourselves", 
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", 
        "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
        "they", "them", "their", "theirs","themselves", "what", "which", "who", 
        "whom", "this", "that", "these", "those", "am", "is", "are", "was", 
        "were", "be", "been", "being", "have", "has", "had", "having", "do", 
        "does", "did", "doing", "will", "would", "can", "could", "may",
        "might", "shall", "ought", "about", "above", "across", "after", 
        "against", "along", "amid", "among", "around", "as", "at", "before", "behind",
        "below", "beneath", "beside", "between", "beyond", "but", "by", 
        "considering", "despite", "down", "during", "except", "for",
        "from", "in", "inside", "into", "like", "near", "next", "notwithstanding",
        "of", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
        "regarding", "round", "since", "than", "through", "throughout", "till", 
        "to", "toward", "towards", "under", "underneath", "unlike", "until", "up",
        "upon", "versus", "via", "with", "within", "without", "cant", "cannot", 
        "couldve", "couldnt", "didnt", "doesnt", "dont", "hadnt", "hasnt", 
        "havent", "hed", "hell", "hes", "howd", "howll", "hows", "id", "ill", 
        "im", "ive", "isnt", "itd", "itll", "its", "lets", "mightve", 
        "shant", "shed", "shell", "shes", 
        "thatll", "thats", "thered", "therell", "therere", "theres", "theyd", 
        "theyll", "theyre", "theyve", "wed", "well", "were", "weve", "werent", 
        "whatd", "whatll", "whatre", "whats", "whatve", "whend", "whenll", 
        "whens", "whered", "wherell", "wheres", "whichd", "whichll", "whichre", 
        "whichs", "whod", "wholl", "whore", "whos", "whove", "whyd", "whyll", 
        "whys", "wont", "wouldve", "wouldnt", "youd", "youll", "youre", "youve",
        "f", "m", "because", "go", "lot", "get", "still", "way", "something", "much",
        "thing", "someone", "person", "anything", "goes", "ok", "so", "just", "mostly", 
        "put", "also", "lots", "yet", "ha", "etc", "even", "one", "bye", "take", "wasnt"]

time = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", 
        "sunday", "morning", "noon", "afternoon", "evening", "night", "midnight",
        "dawn", "dusk", "week", "weekend", "weekends","weekly", "today", 
        "yesterday", "tomorrow", "yesterdays", "todays", "mondays", "tuesdays",
        "wednesdays", "thursdays", "fridays", "saturdays", "sundays", "day",
        "everyday", "daily", "workday", 'time', 'month', 'year', 'pm', 'am', "ago",
        "year", "now"]

reddit = ["welcome", "hi", "hello", "sub", "reddit", "thanks", "thank", "maybe",
          "wo30", "mods", "mod", "moderators", "subreddit", "btw", "aw", "aww", 
          "aww", "hey", "hello", "join", "joined", "post", "rselfimprovement", "blah"]

topic_specific = ["self", "improvement", "change", "action",
    'change', 'start', 'goal', 'habit', 'new', 'old', 
    'care', 'world', 'everyone', 'love', 'u', 'right', 'mean', 'matter',
    'best', 'step', 'focus', 'hard', 'small',
    'bad', 'help', 'time', 'problem', 'issue', 'advice',
    'bit', 'experience', 'different',
    'point', 'situation', 'negative', 'control', 'positive',
    'use', 'question', 'idea', 'amp', 'medium', 'hour', 'day', 'minute',
    'aaaaloot', "selfimprovement", "_", "ampxb"]

stopwords = english + time + reddit + topic_specific

documentAssembler = DocumentAssembler()\
     .setInputCol("cleaned_text")\
     .setOutputCol('document')

tokenizer = Tokenizer() \
            .setInputCols(['document'])\
            .setOutputCol('tokenized')

normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized')

lemmatizer = LemmatizerModel.load("../models/lemma_ewt_en_3.4.3_3.0_1651416655397/")\
      .setInputCols("normalized")\
      .setOutputCol("lemmatized")

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemmatized']) \
     .setOutputCol('words') \
     .setStopWords(stopwords)

from sparknlp.base import Finisher

finisher = Finisher().setInputCols(['words'])

my_pipeline = Pipeline(
      stages = [
          documentAssembler,
          tokenizer,
          normalizer,
          lemmatizer,
          stopwords_cleaner,
          finisher
      ])


pipelineModel = my_pipeline.fit(data)
processed_data = pipelineModel.transform(data)
processed_data.persist()

#Apply TF-IDF filtering
tfizer = CountVectorizer(inputCol='finished_words', outputCol='tf_features', minDF=0.01, vocabSize=1000)
tf_model = tfizer.fit(processed_data)
tf_result = tf_model.transform(processed_data)
vocabulary = tf_model.vocabulary

idfizer = IDF(inputCol='tf_features', outputCol='tf_idf_features')
idf_model = idfizer.fit(tf_result)
tfidf_result = idf_model.transform(tf_result)

processed_data.unpersist()
tfidf_result.persist()

# Asked ChatGPT "how can I filter tokenized words from TF IDF?"

# Define a function to filter words by their TF-IDF score
# UDF to map indices to words using the vocabulary
def filter_tfidf(features, threshold=5, vocabulary=None):
    if features is not None:
        # Filter based on TF-IDF score and map indices to actual words
        return [vocabulary[features.indices[i]] for i in range(len(features.values)) if features.values[i] >= threshold]
    return []

# Register the UDF
filter_udf = udf(lambda features: filter_tfidf(features, threshold=5, vocabulary=vocabulary), ArrayType(StringType()))

# Apply the filtering function
df_filtered_tfidf = tfidf_result.withColumn("filtered_words_tfidf", filter_udf("tf_idf_features"))

tfidf_result.unpersist()

# Create network - Asked chatgpt how to create all possible pair combinations between a list of words

def generate_edges(tokens):
    return [list(pair) for pair in itertools.combinations(tokens, 2)]

generate_edges_udf = udf(generate_edges, ArrayType(ArrayType(StringType())))

df_edges = df_filtered_tfidf.withColumn("edges", generate_edges_udf(F.col("filtered_words_tfidf")))

df_flat_edges = df_edges.select(
    F.col("id"),
    F.explode(F.col("edges")).alias("edge")
)

edges_df = df_flat_edges.select(
    F.col("id").alias("id_doc"),
    F.col("edge")[0].alias("node1"),
    F.col("edge")[1].alias("node2")
)


# Now, need to aggregate edges by count
edges_df = edges_df.withColumn("weight", lit(1))

# Normalize the pairs: ensure node1 is always less than node2, so they can be always on the same order
edges_df = edges_df.withColumn("node1_norm", least(col("node1"), col("node2"))) \
             .withColumn("node2_norm", greatest(col("node1"), col("node2")))

edges_df = edges_df.groupBy("node1_norm", "node2_norm").sum("weight") \
                        .withColumnRenamed("sum(weight)", "weight")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#MODIFY EACH RUN
edges_df.write.mode("overwrite").csv("edges_network5", header=True)
# Create nodes
vertices_df = edges_df.select(F.col("node1_norm").alias("node")).union(edges_df.select(F.col("node2_norm").alias("node"))).distinct()
vertices_df.write.mode("overwrite").csv("nodes_network5", header=True)



