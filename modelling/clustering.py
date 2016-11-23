from modelling.text_processing import split_into_lemmas, split_into_tokens, remove_stop_words

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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import DBSCAN

def create_model(msg_train):

    pipeline = Pipeline([
        ('bow', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
        ('tfidf', TfidfTransformer()),
        ('classifier', model),
    ])

    params = {
        'tfidf__use_idf': (True, False),
        'bow__analyzer': (split_into_lemmas, split_into_tokens, remove_stop_words)
    }

    dbscan =  DBSCAN(pipeline, params)
    return dbscan(msg_train)
