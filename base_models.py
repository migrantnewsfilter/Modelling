from data_processing import merge_mmp
from modelling.models import create_model

from pymongo import MongoClient, ASCENDING

##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
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
    msg_train, msg_test, label_train, label_test = \
        train_test_split(news_feeds_df['text'], news_feeds_df['label'], test_size=0.2)
    model_det = create_model(model, name, msg_train, label_train)
    print model_det.grid_scores_
    print confusion_matrix(label_test, model_det.predict(msg_test))
    print classification_report(label_test, model_det.predict(msg_test))
    with open('serialized_classifiers/' + name +'.pkl', 'wb') as fout:
        cPickle.dump(model_det, fout)
    return model_det

def make_cluster():
    path = './Data/MMP_all_data.csv'
    news_feeds_df = merge_mmp(path)['text']


def write_models_to_disk():
    path = './Data/MMP_all_data.csv'
    estimation(MultinomialNB(), 'NaiveBayes', path)
    estimation(SVC(), 'SVM', path)
    estimation(LogisticRegression(), 'LogReg', path)
    estimation(tree.DecisionTreeClassifier(criterion='gini'), 'DecisionTree', path)
    estimation(RandomForestClassifier(), 'RandomForest', path)
