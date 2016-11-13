#%matplotlib inline
import os
import csv
import json
from bson.json_util import dumps

from pymongo import MongoClient, ASCENDING

##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle

##TEXT MINING PACKAGES
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora, models
from scipy import sparse, io

################################################################################
############################LOADING IN OF DATA SET##############################
################################################################################
#This section pulls the current data from the MongoDB database
#Afterwards the raw json data is filed into a pandas dataframe
#In the end a text variable is created which concatenates the relevant data used in the following text mining exercise

client = MongoClient("mongodb://209.177.92.45:80")

collection = client['newsfilter'].alerts #news - is new one

def get_articles():
    cursor = collection.find().sort('added', ASCENDING)
    return dumps(cursor)

news_feeds_json = get_articles()

#Take a look at raw json
#parsed = json.loads(news_feeds_json)
#print json.dumps(parsed, indent=4, sort_keys=True)

news_feeds = pd.read_json(news_feeds_json)
news_feeds_df = pd.DataFrame(news_feeds)

temp = news_feeds_df['content'].apply(pd.Series)
news_feeds_df = pd.concat([news_feeds_df, temp], axis=1)
news_feeds_df['text'] =  news_feeds_df['title'] + " " + news_feeds_df['body']

news_feeds_df['text'] = news_feeds_df['text'].str.replace("<b>", "", n=-1, case=True, flags=0)
news_feeds_df['text'] = news_feeds_df['text'].str.replace("</b>", "", n=-1, case=True, flags=0)
news_feeds_df['text'] = news_feeds_df['text'].str.replace("&#39", "", n=-1, case=True, flags=0)

################################################################################
##############################OVERVIEW OF DATA SET##############################
################################################################################

#Drop observations that are not labelled - perform unsupervised clustering/labelling?
news_feeds_df = news_feeds_df.dropna()
print  "Size of data set after dropping unlabelled:", len(news_feeds_df)

#DEFINE LENGHT OF TITLE + CONTENT BODY
news_feeds_df['length'] = news_feeds_df['text'].map(lambda text: len(text))

news_feeds_df.length.plot(bins=20, kind='hist')
plt.savefig('Graphical_Analysis/Histogram_Text_Length.png')

news_feeds_df.hist(column='length', by='label', bins=50)
plt.savefig('Graphical_Analysis/Histogram_Text_Length_Label.png')

################################################################################
#################################TEXT PROCESSING################################
################################################################################

execfile("text_processing.py")

#Check Text Processing Functions
print news_feeds_df.text.head().apply(split_into_tokens)
print news_feeds_df.text.head().apply(split_into_lemmas)
print news_feeds_df.text.head().apply(split_into_lemmas).apply(remove_stop_words)
print news_feeds_df.text.head().apply(split_into_lemmas).apply(remove_stop_words).apply(stemming_words)

#NF_df_tokens = news_feeds_df.text.apply(split_into_tokens)
#NF_df_lemmas = news_feeds_df.text.apply(split_into_lemmas)
#NF_df_lemmas_stop = NF_df_lemmas.apply(remove_stop_words)
#Problem with stemming - transforms migration to migrat - Do we want this?
#NF_df_lemmas_stop_stem = NF_df_lemmas_stop.apply(stemming_words)


######################################################
###############VECTOR TRANSFORMATIONS ################
######################################################
#This section applies different transformations to the processed text data
#The goal is to obtain vector representations of frequencies of words in text
#We introduce 2 different transdformations: Simple Bag of Words (bow) and Ngram
#Each vector has as many dimensions as there are unique words (or word combinations) in the text corpus

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(news_feeds_df['text'])
bigram_transformer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

news_feeds_df_bow = bow_transformer.transform(news_feeds_df['text'])
news_feeds_df_bigram = bigram_transformer.fit_transform(news_feeds_df['text']).toarray()

print 'sparse matrix shape - BOW representation:', news_feeds_df_bow.shape #dim: number feeds x unique words
print 'sparse matrix shape - Bigram representation:', news_feeds_df_bigram.shape #dim: number feeds x unique words and 2-word combinations

#More detailed information
#print len(bigram_transformer.vocabulary_)
#print len(bow_transformer.vocabulary_)
#print 'number of non-zeros:', news_feeds_df_bow.nnz
#print 'sparsity: %.2f%%' % (100.0 * news_feeds_df_bow.nnz / (news_feeds_df_bow.shape[0] * news_feeds_df_bow.shape[1]))

######################################################
################tfidf - transformation################
######################################################
#This section implements term weighting and normalization by applying (TF-IDF)
#Choose which dimensionality of the text to use in further analysis (here bigram)

news_feeds_df_textmodel = news_feeds_df_bigram

tfidf_transformer = TfidfTransformer().fit(news_feeds_df_textmodel)
news_feeds_df_tfidf = tfidf_transformer.transform(news_feeds_df_textmodel)

#Should be the same as the previous vectorized representat
print 'sparse matrix shape - TF-IDF representation (bigram):', news_feeds_df_tfidf.shape


######################################################
#######################Outfiling######################
######################################################
#Outfile the current working data

io.mmwrite('Data/data_tfidf_' + time.strftime("%Y_%m_%d") +".mtx", news_feeds_df_tfidf)
news_feeds_df.to_csv('Data/data_df_' + time.strftime("%Y_%m_%d") +".csv", sep='\t', encoding='utf-8')
