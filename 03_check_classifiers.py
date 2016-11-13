execfile("text_processing.py")
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

svm_detector = cPickle.load(open('serialized_classifiers/svm_news_classifier.pkl'))
nb_detector = cPickle.load(open('serialized_classifiers/nb_news_classifier.pkl'))
logistic_detector = cPickle.load(open('serialized_classifiers/logistic_news_classifier.pkl'))
dec_tree_detector = cPickle.load(open('serialized_classifiers/dec_tree_news_classifier.pkl'))
randomF_detector = cPickle.load(open('serialized_classifiers/random_forest_news_classifier.pkl'))

print "Naive Bayes", nb_detector.predict_proba(["libyan president mourns fact that migrants die in the mediterranean"])[0], nb_detector.predict(["libyan president mourns fact that migrants die in the mediterranean"])[0]
print "Naive Bayes", nb_detector.predict_proba(["migrant shot dead while trying to cross border into texas"])[0], nb_detector.predict(["migrant shot dead while trying to cross border into texas"])[0]

print "SVM", svm_detector.predict(["libyan president mourns fact that migrants die in the mediterranean"])[0]
print "SVM", svm_detector.predict(["migrant shot dead while trying to cross border into texas"])[0]

print "Logistic Regression", logistic_detector.predict(["libyan president mourns fact that migrants die in the mediterranean"])[0]
print "Logistic Regression", logistic_detector.predict(["migrant shot dead while trying to cross border into texas"])[0]

print "Decision Tree", dec_tree_detector.predict(["libyan president mourns fact that migrants die in the mediterranean"])[0]
print "Decision Tree", dec_tree_detector.predict(["migrant shot dead while trying to cross border into texas"])[0]

print "Random Forest", randomF_detector.predict(["libyan president mourns fact that migrants die in the mediterranean"])[0]
print "Random Forest", randomF_detector.predict(["migrant shot dead while trying to cross border into texas"])[0]
