import sparknlp
from sparknlp.annotator import Tokenizer, PerceptronModel
from sparknlp.base import DocumentAssembler
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA
from sparknlp.annotator import StopWordsCleaner
from pyspark.sql import types as T
from pyspark.sql import functions as F
from sparknlp.annotator import NGramGenerator
from sparknlp.base import Finisher
from pyspark.ml.tuning import ParamGridBuilder

spark = sparknlp.start()

og_data = spark.read.csv("../data_topicmodel.csv", header= True).select(["id", "cleaned_text"])

#Remove sample
sample_data = og_data.sample(0.001)

#Preprocessing
documentAssembler = DocumentAssembler()\
     .setInputCol("cleaned_text")\
     .setOutputCol('document')

tokenizer = Tokenizer() \
            .setInputCols(['document'])\
            .setOutputCol('tokenized')

normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') 

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
        "does", "did", "doing", "will", "would", "should", "can", "could", "may",
        "might", "must", "shall", "ought", "about", "above", "across", "after", 
        "against", "along", "amid", "among", "around", "as", "at", "before", "behind",
        "below", "beneath", "beside", "between", "beyond", "but", "by", 
        "concerning", "considering", "despite", "down", "during", "except", "for",
        "from", "in", "inside", "into", "like", "near", "next", "notwithstanding",
        "of", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
        "regarding", "round", "since", "than", "through", "throughout", "till", 
        "to", "toward", "towards", "under", "underneath", "unlike", "until", "up",
        "upon", "versus", "via", "with", "within", "without", "cant", "cannot", 
        "couldve", "couldnt", "didnt", "doesnt", "dont", "hadnt", "hasnt", 
        "havent", "hed", "hell", "hes", "howd", "howll", "hows", "id", "ill", 
        "im", "ive", "isnt", "itd", "itll", "its", "lets", "mightve", "mustve", 
        "mustnt", "shant", "shed", "shell", "shes", "shouldve", "shouldnt", 
        "thatll", "thats", "thered", "therell", "therere", "theres", "theyd", 
        "theyll", "theyre", "theyve", "wed", "well", "were", "weve", "werent", 
        "whatd", "whatll", "whatre", "whats", "whatve", "whend", "whenll", 
        "whens", "whered", "wherell", "wheres", "whichd", "whichll", "whichre", 
        "whichs", "whod", "wholl", "whore", "whos", "whove", "whyd", "whyll", 
        "whys", "wont", "wouldve", "wouldnt", "youd", "youll", "youre", "youve",
        "f", "m", "because", "go", "lot", "get", "still", "way", "something", "much",
        "thing", "someone", "person", "anything", "goes", "ok", "so", "just", "mostly", 
        "put", "also", "lots", "yet", "ha", "etc"]

time = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", 
        "sunday", "morning", "noon", "afternoon", "evening", "night", "midnight",
        "dawn", "dusk", "week", "weekend", "weekends","weekly", "today", 
        "yesterday", "tomorrow", "yesterdays", "todays", "mondays", "tuesdays",
        "wednesdays", "thursdays", "fridays", "saturdays", "sundays", "day",
        "everyday", "daily", "workday", 'time', 'month', 'year', 'pm', 'am', "ago",
        "year"]

reddit = ["welcome", "hi", "hello", "sub", "reddit", "thanks", "thank", "maybe",
          "wo30", "mods", "mod", "moderators", "subreddit", "btw", "aw", "aww", 
          "aww", "hey", "hello", "join", "joined", "post", "rselfimprovement"]

topic_specific = ["self", "improvement", "change", "action",
    'change', 'start', 'goal', 'habit', 'new', 'old', 
    'care', 'world', 'everyone', 'love', 'u', 'right', 'mean', 'matter',
    'best', 'step', 'focus', 'hard', 'small',
    'bad', 'help', 'time', 'problem', 'issue', 'advice',
    'bit', 'experience', 'different',
    'point', 'situation', 'negative', 'control', 'positive',
    'use', 'question', 'idea', 'amp', 'medium', 'hour', 'day', 'minute',
    'aaaaloot', "selfimprovement", "_"]

stopwords = english + time + reddit + topic_specific

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['normalized']) \
     .setOutputCol('unigrams') \
     .setStopWords(stopwords)

pos = PerceptronModel.load("/project/macs40123/spark-jars/pos_anc_en_3.0.0_3.0_1614962126490/")\
      .setInputCols("document", "unigrams")\
      .setOutputCol("pos")

finisher = Finisher().setInputCols(['unigrams', 'pos'])

my_pipeline = Pipeline(
      stages = [
          documentAssembler,
          tokenizer,
          normalizer,
          stopwords_cleaner,
          #ngrammer,
          pos,
          finisher
      ])

pipelineModel = my_pipeline.fit(sample_data)
processed_data = pipelineModel.transform(sample_data)

#Filter by POS
def filter_unigrams(finished_unigrams, finished_pos):
    '''Filters individual words based on their POS tag'''
    return [word for word, pos in zip(finished_unigrams, finished_pos)
            if pos in ['JJ', 'NN', 'NNS', 'NNPS']]

udf_filter_unigrams = F.udf(filter_unigrams, T.ArrayType(T.StringType()))

processed_data = processed_data.withColumn('filtered_unigrams_by_pos', udf_filter_unigrams(
                                                   F.col('finished_unigrams'),
                                                   F.col('finished_pos')))

#Now that POS was done, lemmatization makes more sense at this point

#Merge tokens as just one string to be able to take it as a document in the new Pipeline
tokens_as_string = F.udf(lambda x: ' '.join(x), T.StringType())
processed_data = processed_data.withColumn('joined_tokens', tokens_as_string(F.col('filtered_unigrams_by_pos')))

last_documentAssembler = DocumentAssembler() \
     .setInputCol('joined_tokens') \
     .setOutputCol('joined_document')

last_tokenizer = Tokenizer() \
     .setInputCols(['joined_document']) \
     .setOutputCol('tokenized')
     
lemmatizer = LemmatizerModel.load("../models/lemma_ewt_en_3.4.3_3.0_1651416655397/")\
      .setInputCols("tokenized")\
      .setOutputCol("lemmatized")

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemmatized']) \
     .setOutputCol('final') \
     .setStopWords(stopwords)

last_finisher = Finisher() \
     .setInputCols(['final']) \

last_pipeline = Pipeline() \
     .setStages([last_documentAssembler,                  
                 last_tokenizer,
                 lemmatizer,
                 stopwords_cleaner,
                 last_finisher])

final_data = last_pipeline.fit(processed_data).transform(processed_data)

## Vectorization
#Apply TF-IDF filtering
tfizer = CountVectorizer(inputCol='finished_final', outputCol='tf_features', minDF=0.01, maxDF=0.80)
tf_model = tfizer.fit(final_data)
tf_result = tf_model.transform(final_data)

idfizer = IDF(inputCol='tf_features', outputCol='tf_idf_features')
idf_model = idfizer.fit(tf_result)
tfidf_result = idf_model.transform(tf_result)

## LDA
lda = LDA(featuresCol='tf_idf_features', seed=2503)

paramGrid = ParamGridBuilder() \
    .addGrid(lda.k, [5, 8]) \
    .addGrid(lda.maxIter, [10, 20]) \
    .build()

def evaluate_model(model, data):
    log_likelihood = model.logLikelihood(data)
    perplexity = model.logPerplexity(data)
    return log_likelihood, perplexity

best_perplexity = 100
best_log_likelihood = -99999999999
best_param_map_ll = None
best_param_map_pxty = None

for param_map in paramGrid:
    model = lda.copy(param_map).fit(tfidf_result)
    log_likelihood, perplexity = evaluate_model(model, tfidf_result)
    print(f"Params: {param_map}, Log Likelihood: {log_likelihood}, Perplexity: {perplexity}")
    if perplexity < best_perplexity: 
        best_perplexity = perplexity
        best_param_map_pxty = param_map
    if log_likelihood > best_log_likelihood: 
        best_log_likelihood = log_likelihood
        best_param_map_ll = param_map

print("Best parameters for perplexity")
print(best_param_map_pxty)
print(best_perplexity)
print("------------------------------------------")
print("Best parameters for log likelihood")
print(best_param_map_ll)
print(best_log_likelihood)