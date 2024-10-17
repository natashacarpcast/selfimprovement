#https://sparknlp.org/api/python/reference/autosummary/sparknlp/annotator/embeddings/doc2vec/index.html#sparknlp.annotator.embeddings.doc2vec.Doc2VecModel

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
embeddings = Doc2VecModel.pretrained() \
    .setInputCols(["token"]) \
    .setOutputCol("embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
])
data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(1, 80)