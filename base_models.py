from text_processing import *
from data_processing import *

##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle

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
###################FIT/PARAMETER TUNING - NAIVE BAYES###########################
################################################################################

def estimation(model, name, path):
    news_feeds_df = merge_mmp(path)
    ##
    msg_train, msg_test, label_train, label_test = \
        train_test_split(news_feeds_df['text'], news_feeds_df['label'], test_size=0.2)
    ##
    print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)
    pipeline = Pipeline([
        ('bow', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
        ('tfidf', TfidfTransformer()),
        ('classifier', model),
    ])
    # Does the extra processing cost of lemmatization (vs. just plain words) really help
    if model == 'SVC':
        param_svm = [
          {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
          {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
        ]
    else:
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
    model_det = grid.fit(msg_train, label_train)
    print model_det.grid_scores_
    print confusion_matrix(label_test, model_det.predict(msg_test))
    print classification_report(label_test, model_det.predict(msg_test))
    with open('serialized_classifiers/' + name +'.pkl', 'wb') as fout:
        cPickle.dump(model_det, fout)
    return model_det

path = './Data/MMP_all_data.csv'

estimation(MultinomialNB(), 'NaiveBayes', path)
estimation(SVC(), 'SVM', path)
estimation(LogisticRegression(), 'LogReg', path)
estimation(tree.DecisionTreeClassifier(criterion='gini'), 'DecisionTree', path)
estimation(RandomForestClassifier(), 'RandomForest', path)
