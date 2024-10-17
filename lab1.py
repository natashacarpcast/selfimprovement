#Sources
#https://github.com/apache/spark/blob/master/examples/src/main/python/ml/tf_idf_example.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer
from pyspark.ml.feature import MinHashLSH

spark = SparkSession \
        .builder \
        .appName("lab1") \
        .getOrCreate()

#Read CSV file 
df = spark.read.option("header", True).csv("just_text_submissions.csv")

#Tokenize
tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokenized")
tokenized_df = tokenizer.transform(df)

#Vectorize
cv = CountVectorizer(inputCol="tokenized", outputCol="vectorized")
model_cv = cv.fit(tokenized_df)
vectorized_df = model_cv.transform(tokenized_df)

#TF-IDF vectors
idf = IDF(inputCol="vectorized", outputCol="tf-idf", minDocFreq = 10)
model_idf = idf.fit(vectorized_df)
weighted_df = model_idf.transform(vectorized_df)
weighted_df.show()

# Instantiate minhashing model
mh = MinHashLSH()
mh.setInputCol("tf-idf")
mh.setOutputCol("hashes")
mh.setSeed(2503)

# Fit model on tf-idf vectors
model = mh.fit(weighted_df)
model.setInputCol("tf-idf")

# Create a test item
test = spark.createDataFrame([("test01", "this inner battle went on for months now i m sitting here writing the structure for a paper i have to write that will basicly determine how my life will go on in the next few years    about 2 3 months ago i decided to quit my job and attend school again  after one year on that school  successfully finishing it  i will have a certificate report that allows me to study at a college of my choise  normally that would take up to 2 3 years  but i took the shorter and harder way  because i m 26 years old and don t want to lose so much time     i have dedicated almost 6 7 years of my life to computer science and i wanted to study something that would be called  economic computer science  in english  but now i somehow can t imagine doing it anymore  i don t think what i do is valueable  i m just moving numbers data around  even in  business intelligence   where you are basicly solving complicated problems all day and should never get bored   i just can t see myself doing this all my life  but on the other hand  it is just work  my life happens while i don t work     and now there is the thing with my heart  i really like thinking about morals  ethics  about what is right or wrong  i like philosophy  psychology and stuff  i like to read literature about insights  epistemology   i think if i do something with this  i will do something valueable with whatever i will be working  philosophy basicly shaped all our life and psychology has the power to cure minds      all these thoughts were burried deep inside my soul  i burried them because i wanted to stop thinking so much about stuff and instead wanted experience life  but i was really passionate about all this  i engaged in countless discussions and even made some good friends while pursuing my interests   and now the girl i m seeing has brought back all those thoughts  we engage in countless discussions about truth  self development  politics  education  pedagogy  we talk for hours  i even went on a course with her at her university and i really liked it     i also have read more books in the last 2 3 months than in my entire life  i would say that i m passionate about all this  and because i read so much and thought so much about all this  i think it is important to follow ones passion     but what if        what if this is just a phase    what if i m throwing away my life     what if       what if        the title of my first paper is  translated into english      the pursuit of truth and unity with oneself and the world a look at the self discovery process based on the novel  siddhartha  by hermann hesse   the title of my second paper is     development of a data warehouse based on the sales process of company xxx     all those thoughts and i still can t decide what to do  i m almost paralyzed     it would appreciate to read about your thoughts  thanks ")], ["id", "cleaned_text"])
tokenized_test = tokenizer.transform(test)
vectorized_test = model_cv.transform(test)
weighted_test = model_idf.transform(test)
weighted_test.show()
