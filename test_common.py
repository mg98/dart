from tfidf import TFIDF
from bm25 import BM25

corpus = {
    "doc1": "This is a sample document about cats and dogs and so",
    "doc2": "Another example document discussing machine learning",
    "doc3": "A third document about natural language processing",
    "doc4": "Sample text about dogs and their training"
}

tfidf = TFIDF(corpus)
print(tfidf.get_tf_idf("doc1", "okay"))
print(tfidf.get_tf_idf("doc1", "cats"))
print(tfidf.get_tf_idf("doc1", "document"))
print(tfidf.get_tf_idf("doc1", "and"))
