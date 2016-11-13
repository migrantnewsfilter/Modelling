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
from scipy import sparse, io

##TEXT MINING PACKAGES
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import sklearn
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


################################################################################
####################################DATA SET UP#################################
################################################################################

news_feeds_df_tfidf = io.mmread('Data/data_tfidf_' + time.strftime("%Y_%m_%d") +".mtx")
news_feeds_df_tfidf = news_feeds_df_tfidf.tocsr()
news_feeds_df_tfidf = news_feeds_df_tfidf.toarray()

news_feeds_df = pd.DataFrame.from_csv('Data/data_df_' + time.strftime("%Y_%m_%d") +".csv", sep='\t', encoding='utf-8')
#news_feeds_df = news_feeds_df.toarray()

msg_train, msg_test, label_train, label_test = \
    train_test_split(news_feeds_df['text'], news_feeds_df['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

####First Naive Bayes Model - PLAIN - USING ALL DATA - ONLY INSAMPLE FIT
#Naive Bayes using scikit-learn
#%time spam_detector = MultinomialNB().fit(news_feeds_df_tfidf, news_feeds_df['label'])

#all_predictions = spam_detector.predict(news_feeds_df_tfidf)

#print 'accuracy', accuracy_score(news_feeds_df['label'], all_predictions)
#print 'confusion matrix\n', confusion_matrix(news_feeds_df['label'], all_predictions)

#plt.matshow(confusion_matrix(news_feeds_df['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
#plt.title('confusion matrix')
#plt.colorbar()
#plt.ylabel('expected label')
#plt.xlabel('predicted label')
#plt.savefig('Confusion_Matrix.png')

#print classification_report(news_feeds_df['label'], all_predictions)


################################################################################
###################FIT/PARAMETER TUNING - NAIVE BAYES###########################
################################################################################

pipeline = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# Does the extra processing cost of lemmatization (vs. just plain words) really help
params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens, remove_stop_words),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

nb_detector = grid.fit(msg_train, label_train)
print nb_detector.grid_scores_
print confusion_matrix(label_test, nb_detector.predict(msg_test))
print classification_report(label_test, nb_detector.predict(msg_test))

################################################################################
###################FIT/PARAMETER TUNING - SVM###################################
################################################################################

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
print svm_detector.grid_scores_
print confusion_matrix(label_test, svm_detector.predict(msg_test))
print classification_report(label_test, svm_detector.predict(msg_test))


################################################################################
###################FIT/PARAMETER TUNING - LOGISTIC##############################
################################################################################

pipeline_log = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression()),
])

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid_log = GridSearchCV(
    pipeline_log,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

logistic_detector = grid_log.fit(msg_train, label_train)
print logistic_detector.grid_scores_
print confusion_matrix(label_test, logistic_detector.predict(msg_test))
print classification_report(label_test, logistic_detector.predict(msg_test))

################################################################################
#####################PARAMETER TUNING - DECISION TREE###########################
################################################################################

pipeline_dec_tree = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('classifier', tree.DecisionTreeClassifier(criterion='gini')),
])

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid_dec_tree = GridSearchCV(
    pipeline_dec_tree,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

dec_tree_detector = grid_dec_tree.fit(msg_train, label_train)
print dec_tree_detector.grid_scores_
print confusion_matrix(label_test, dec_tree_detector.predict(msg_test))
print classification_report(label_test, dec_tree_detector.predict(msg_test))

################################################################################
#####################PARAMETER TUNING - RANDOM FOREST###########################
################################################################################

pipeline_RF = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier()),
])

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid_RF = GridSearchCV(
    pipeline_RF,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

randomF_detector = grid_RF.fit(msg_train, label_train)
print randomF_detector.grid_scores_
print confusion_matrix(label_test, randomF_detector.predict(msg_test))
print classification_report(label_test, randomF_detector.predict(msg_test))


################################################################################
########################OUTFILING  PREDICTORS###################################
################################################################################

# store the news classifiers to disk after training
with open('serialized_classifiers/svm_news_classifier.pkl', 'wb') as fout:
    cPickle.dump(svm_detector, fout)

with open('serialized_classifiers/nb_news_classifier.pkl', 'wb') as fout:
    cPickle.dump(nb_detector, fout)

with open('serialized_classifiers/logistic_news_classifier2.pkl', 'wb') as fout:
    cPickle.dump(logistic_detector, fout)

with open('serialized_classifiers/dec_tree_news_classifier.pkl', 'wb') as fout:
    cPickle.dump(dec_tree_detector, fout)

with open('serialized_classifiers/random_forest_news_classifier.pkl', 'wb') as fout:
    cPickle.dump(randomF_detector, fout)
