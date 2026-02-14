Gutenberg Books Analysis with PySpark

This project performs text analysis on Project Gutenberg books using PySpark. It extracts metadata, cleans text, computes TF-IDF features, and calculates cosine similarity between books.

Features

Load multiple books from a folder.

Extract metadata: Title, Author, Release Date, Language, Encoding.

Clean text by removing Gutenberg headers, punctuation, and lowercasing.

Tokenize and remove stopwords.

Compute TF-IDF vectors for each book.

Compute cosine similarity between books for similarity analysis.

Extract publication year for statistical analysis.

Requirements

Python 3.10+

PySpark 3.x

NumPy

Install Python dependencies:

pip install pyspark numpy

Usage

Place your Project Gutenberg books in a folder.

Update the books_folder path in GutenbergAnalysis.py.

Run the script:

spark-submit GutenbergAnalysis.py


The script will:

Show extracted metadata.

Clean the text and compute TF-IDF features.

Compute similarity with the first book (example).

Show yearly distribution of books.

Output

Metadata and cleaned text for each book.

TF-IDF feature vectors.

Cosine similarity between books.

Year-wise count of books.

Example
+---------+--------------------+----+
|file_name|              title |year|
+---------+--------------------+----+
| 101.txt | Hacker Crackdown    |1994|
| 102.txt | Pudd'nhead Wilson   |1994|
| 103.txt | Around the World... |2008|
+---------+--------------------+----+