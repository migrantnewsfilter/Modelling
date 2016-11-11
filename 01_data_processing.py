%matplotlib inline
import os
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas as pd
import sklearn
import cPickle
import json
import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from pymongo import MongoClient, ASCENDING
from bson.json_util import dumps
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

################################################################################
############################LOADING IN OF DATA SET##############################
################################################################################
os.chdir('..')
client = MongoClient("mongodb://209.177.92.45:80")

collection = client['newsfilter'].alerts

def get_articles():
    print 'get articles!!!'
    cursor = collection.find().sort('added', ASCENDING)
    return dumps(cursor)

news_feeds_json = get_articles()
parsed = json.loads(news_feeds_json)
print json.dumps(parsed, indent=4, sort_keys=True)

news_feeds = pd.read_json(news_feeds_json)
news_feeds_df = pd.DataFrame(news_feeds)

temp = news_feeds_df['content'].apply(pd.Series)
news_feeds_df = pd.concat([news_feeds_df, temp], axis=1)
news_feeds_df['text'] =  news_feeds_df['title'] + " " + news_feeds_df['body']

################################################################################
##############################OVERVIEW OF DATA SET##############################
################################################################################

#Drop observations that are not labelled
news_feeds_df = news_feeds_df.dropna()
print  "Size of data set:", len(news_feeds_df)

#DEFINE LENGHT OF TITLE + CONTENT BODY
news_feeds_df['length'] = news_feeds_df['text'].map(lambda text: len(text))

news_feeds_df.length.plot(bins=20, kind='hist')
plt.savefig('Histogram_Text_Length.png')

news_feeds_df.hist(column='length', by='label', bins=50)
plt.savefig('Histogram_Text_Length_Label.png')

################################################################################
#################################DATA PROCESSING################################
################################################################################

def split_into_tokens(text):
    #split a message into its individual words
    #text = unicode(text, 'utf8') - convert bytes into proper unicode - does not work because already unicode
    return TextBlob(text).words

def split_into_lemmas(text):
    #normalize words into their base form (lemmas)
    text = text.lower()
    words = TextBlob(text).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

news_feeds_df.text.head().apply(split_into_tokens)
news_feeds_df.text.head().apply(split_into_lemmas)

######################################################
###############CONVERT TOKENS TO VECTOR###############
######################################################

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(news_feeds_df['text'])
#Each vector has as many dimensions as there are unique words in the text corpus
print len(bow_transformer.vocabulary_)
news_feeds_df_bow = bow_transformer.transform(news_feeds_df['text'])
print 'sparse matrix shape:', news_feeds_df_bow.shape #dim: number feeds x unique words
print 'number of non-zeros:', news_feeds_df_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * news_feeds_df_bow.nnz / (news_feeds_df_bow.shape[0] * news_feeds_df_bow.shape[1]))

######################################################
################tfidf - transformation################
######################################################


#Term weighting and normalization can be done with TF-IDF
tfidf_transformer = TfidfTransformer().fit(news_feeds_df_bow)
#transform the entire bag-of-words corpus into TF-IDF corpus at once:
news_feeds_df_tfidf = tfidf_transformer.transform(news_feeds_df_bow)
print news_feeds_df_tfidf.shape
