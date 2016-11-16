##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from base_models import *
##LOAD PREDICTORS

estimation(MultinomialNB(), 'NaiveBayes', path)
estimation(SVC(), 'SVM', path)
estimation(LogisticRegression(), 'LogReg', path)
estimation(tree.DecisionTreeClassifier(criterion='gini'), 'DecisionTree', path)
estimation(RandomForestClassifier(), 'RandomForest', path)

pred_models = {
    'SVM': cPickle.load(open('./serialized_classifiers/SVM.pkl')),
    'Naive_Bayes':  cPickle.load(open('./serialized_classifiers/NaiveBayes.pkl')),
    'Logistic_Regression': cPickle.load(open('./serialized_classifiers/LogReg.pkl')),
    'Decision_Tree': cPickle.load(open('./serialized_classifiers/DecisionTree.pkl')),
    'Random_Forest': cPickle.load(open('./serialized_classifiers/RandomForest.pkl'))
}


##LOAD FAKE TEST DATA SET

test_data = pd.DataFrame.from_csv('./Data/test_data.csv', sep='\t', encoding='utf-8')
list(test_data.columns.values)

def test_pred_mat_plot(predictor, name):
    all_predictions = predictor.predict(test_data['text'])
    print name
    print 'accuracy', accuracy_score(test_data['label'], all_predictions)
    print 'confusion matrix\n', confusion_matrix(test_data['label'], all_predictions)
    plt.matshow(confusion_matrix(test_data['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
    plt.title('confusion matrix ' + name)
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    plt.savefig('/Users/robertlange/Desktop/news_filter_project/Modelling/Graphical_Analysis/Confusion_' + name + '.png')

for name, predictor in pred_models.iteritems():
    test_pred_mat_plot(predictor, name)

#print "Naive Bayes", nb_detector.predict_proba(["libyan president mourns fact that migrants die in the mediterranean"])[0], nb_detector.predict(["libyan president mourns fact that migrants die in the mediterranean"])[0]
#print "Naive Bayes", nb_detector.predict_proba(["migrant shot dead while trying to cross border into texas"])[0], nb_detector.predict(["migrant shot dead while trying to cross border into texas"])[0]
