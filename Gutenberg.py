# GutenbergAnalysis.py
import os
import re
import numpy as np
from pyspark.sql import SparkSession, Row, functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.linalg import DenseVector, SparseVector

# ----------------------------
# Spark session
# ----------------------------
spark = SparkSession.builder.appName("GutenbergAnalysis").getOrCreate()
sc = spark.sparkContext

# ----------------------------
# Functions
# ----------------------------
def extract_metadata(text):
    title = re.search(r"Title:\s*(.*)", text)
    author = re.search(r"Author:\s*(.*)", text)
    release_date = re.search(r"Release Date:\s*(.*)", text)
    language = re.search(r"Language:\s*(.*)", text)
    encoding = re.search(r"Character set encoding:\s*(.*)", text)
    return (
        title.group(1).strip() if title else None,
        author.group(1).strip() if author else None,
        release_date.group(1).strip() if release_date else None,
        language.group(1).strip() if language else None,
        encoding.group(1).strip() if encoding else None
    )

def clean_text(text):
    text = re.sub(r"(\*{3} START OF THIS PROJECT GUTENBERG EBOOK .* \*{3})", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\*{3} END OF THIS PROJECT GUTENBERG EBOOK .* \*{3})", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def cosine_similarity(vec1, vec2):
    v1 = vec1.toArray() if isinstance(vec1, (DenseVector, SparseVector)) else vec1
    v2 = vec2.toArray() if isinstance(vec2, (DenseVector, SparseVector)) else vec2
    dot = float(np.dot(v1, v2))
    norm1 = float(np.sqrt(np.dot(v1, v1)))
    norm2 = float(np.sqrt(np.dot(v2, v2)))
    return float(dot / (norm1 * norm2)) if norm1 != 0 and norm2 != 0 else 0.0

# ----------------------------
# Load books
# ----------------------------
books_folder = "/mnt/d/downloads/D184MB/D184MB"  # replace with your path
books_rdd = sc.wholeTextFiles(books_folder)
books_df = books_rdd.map(lambda x: Row(file_name=os.path.basename(x[0]), text=x[1])).toDF()

# ----------------------------
# Extract metadata & clean text
# ----------------------------
metadata_udf = F.udf(lambda x: extract_metadata(x),
                     returnType=F.StructType()
                     .add("title", StringType())
                     .add("author", StringType())
                     .add("release_date", StringType())
                     .add("language", StringType())
                     .add("encoding", StringType()))

clean_text_udf = F.udf(clean_text, StringType())

books_df = books_df.withColumn("metadata", metadata_udf("text")) \
                   .select(
                       "file_name",
                       F.col("metadata.title"),
                       F.col("metadata.author"),
                       F.col("metadata.release_date"),
                       F.col("metadata.language"),
                       F.col("metadata.encoding"),
                       "text"
                   )

books_df = books_df.withColumn("clean_text", clean_text_udf("text"))

# ----------------------------
# Extract year
# ----------------------------
books_df = books_df.withColumn(
    "year",
    F.when(F.col("release_date").rlike(r"\d{4}"),
           F.regexp_extract("release_date", r"(\d{4})", 1).cast(IntegerType())
           ).otherwise(None)
)

# ----------------------------
# TF-IDF
# ----------------------------
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
words_df = tokenizer.transform(books_df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_df = remover.transform(words_df)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
featurized_df = hashingTF.transform(filtered_df)

idf = IDF(inputCol="rawFeatures", outputCol="tfidfFeatures")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

# ----------------------------
# Compute cosine similarity with first book (example)
# ----------------------------
target_vector = tfidf_df.select("tfidfFeatures").first()[0]
cosine_udf = F.udf(lambda x: cosine_similarity(target_vector, x), DoubleType())
similarity_df = tfidf_df.withColumn("similarity", cosine_udf(F.col("tfidfFeatures"))) \
                        .select("file_name", "title", "author", "year", "similarity")

# ----------------------------
# Show results
# ----------------------------
books_df.show(5, truncate=50)
similarity_df.orderBy(F.desc("similarity")).show(5, truncate=50)
tfidf_df.select("file_name", "tfidfFeatures").show(5)
books_df.groupBy("year").count().orderBy("year").show()
