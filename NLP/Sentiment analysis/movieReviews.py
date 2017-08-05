import os
from sklearn.feature_extraction import text
from sklearn import ensemble
import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk import corpus

os.chdir("D:/Deep Analytics/NLP/Sentiment analysis")

# clean-up text
def preprocess_review(review):
    # remove HTML tags
    review_text = BeautifulSoup(review).get_text()
    # remove non-numbers (such as punctuations)
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # convert to lower case
    return review_text.lower()
    
# split sentences into words
def tokenize(review):
    return review.split()
    
# Download input file labeledTrainData.tsv from https://www.kaggle.com/c/word2vec-nlp-tutorial/data

# Read reviews from file
# 'quoting' used to ignore the " in review (and not treat it as end of string)
movie_train = pd.read_csv("labeledTrainData.tsv", header = 0,
                          delimiter = "\t", quoting = 3)
movie_train.shape # (25000, 3) ==> 25000 rows and 3 columns
movie_train.info()
movie_train.loc[0:4,'review']

# 1st raw review
review_tmp = movie_train['review'][0]
# remove HTML tags from raw review
review_tmp = BeautifulSoup(review_tmp).get_text()
# everything other than alphabets is replaced by ' '.
review_tmp = re.sub("[^a-zA-Z]", " ", review_tmp)
# split cleaned-up review into words
review_tmp_words = review_tmp.split()

# Bag-of-words model built using one of 3 different types of vectorizers - count, TF-IDF, hash
# The model produces a numeric matrix known as document-term matrix or term-document matrix.
# It is a mathematical matrix that describes the frequency of terms that occur in a collection
# of documents. In a document-term matrix, rows correspond to documents in the collection and
# columns correspond to terms.

# Count vectorizer. Stores the count of each word in text.
# Stop words - commonly used words such as 'this', 'while' etc. are removed
vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  tokenizer = tokenize,    \
                                  stop_words = 'english',   \
                                  max_features = 5000)
vectorizer.get_stop_words() # list of stop words 

# Apply vectorizer to text data, and get a copy of the term document matrix
# This matrix is a bag-of-words
features = vectorizer.fit_transform(movie_train.loc[0:3,'review']).toarray()

# Below calls to vectorizer work only after it is applied on text data
vocab = vectorizer.get_feature_names() # list of words in text data
vectorizer.vocabulary_ # list of words and corresponding indexes in text data

# Make a column-wise sum. Gives the count of each word across input reviews
word_count = np.sum(features, axis=0)

# Combine the list of words and their counts into a single data-frame; display them
for tag, count in zip(vocab, word_count):
    print(tag, count)

# Create a random forest classifier    
forest = ensemble.RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest.fit(features, movie_train.loc[0:3,'sentiment'])

forest = forest.fit(features, movie_train['sentiment'])