#https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.feature.Word2Vec.html
#https://www.bigspark.dev/word-embedding-word2vec-algorithm-implementation-using-sparkmllib/


from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Word2Vec

spark = SparkSession \
        .builder \
        .appName("lab1") \
        .getOrCreate()

#Read CSV file 
df = spark.read.option("header", True).csv("just_text_submissions.csv")

#Tokenize
tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokenized")
tokenized_df = tokenizer.transform(df)

#Remove stopwords 
#Asked ChatGPT: 
#Write me a stop words python list, all lower case, including standard english words and also reddit specific, such as URLs, etc.

stop_words = [
    # Standard English stop words
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "are not", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can not", "cannot", "could", 
    "could not", "did", "did not", "do", "does", "does not", "doing", "do not", "down", "during", "each", "few", "for", 
    "from", "further", "had", "had not", "has", "has not", "have", "have not", "having", "he", "he s", "he would", 
    "he will", "he s", "her", "here", "here s", "hers", "herself", "him", "himself", "his", "how", "how s", "i", 
    "i would", "i will", "i m", "i have", "if", "in", "into", "is", "is not", "it", "it s", "its", "itself", "let us", 
    "me", "more", "most", "must not", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", 
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shall not", "she", "she s", 
    "she would", "she will", "she s", "should", "should not", "so", "some", "such", "than", "that", "that s", "the", 
    "their", "theirs", "them", "themselves", "then", "there", "there s", "these", "they", "they would", "they will", 
    "they are", "they have", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "was not", 
    "we", "we would", "we will", "we are", "we have", "were", "were not", "what", "what s", "when", "when s", 
    "where", "where s", "which", "while", "who", "who s", "whom", "why", "why s", "with", "will not", "would", 
    "would not", "you", "you would", "you will", "you are", "you have", "your", "yours", "yourself", "yourselves",
    
    # Reddit-specific stop words
    "r", "u", "reddit", "subreddit", "post", "posted", "comment", "thread", "upvote", "downvote", "karma", "flair", 
    "mods", "moderator", "nsfw", "ama", "op", "tldr", "edit", "title", "link", "url", "http", "https", "www", 
    "com", "org", "net", "jpg", "png", "gif", "imgur", "youtube", "youtu", "sub", "user", "self", "removed", 
    "deleted", "please", "thanks", "thank", "hi", "hello", "help", "lol", "btw", "imo", "imho", "fyi", "anyone", 
    "everyone",
    
    # Common fillers and additional terms
    "just", "like", "know", "get", "got", "make", "go", "going", "think", "see", "say", "said", "would", "one", 
    "two", "three", "really", "actually", "thing", "things", "people", "even", "way", "something", "anything", 
    "everything", "maybe", "someone", "anywhere", "everyone", "everybody", "someone", "anyone", "thing", "things"]
remover = StopWordsRemover(stopWords=stop_words, inputCol="tokenized", 
                                         outputCol="cleaned_tokens")
cleaned_df = remover.transform(tokenized_df)

#Word2Vec
word2Vec = Word2Vec(vectorSize=100, seed =2503, minCount=10, inputCol="cleaned_tokens", outputCol="model")
model = word2Vec.transform(cleaned_df)
model.findSynonyms("morality",10)