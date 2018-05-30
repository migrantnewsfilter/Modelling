import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, LeaveOneGroupOut
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from modelling.utils import clean_html, preprocessor, get_articles
from datetime import datetime, timedelta


def vectorizer(dat, model = TfidfVectorizer, preprocessor=preprocessor):
    """ Creates a tfidf vectorizer from an array-like of strings """
    return model(
        stop_words = 'english',
        ngram_range = (1,3),
        preprocessor = preprocessor
    ).fit(dat)

def get_errors(X_test, y_test, preds):
    df = pd.DataFrame({'text': X_test, 'prediction': preds, 'label': y_test})
    problems = df[df.label != df.prediction]
    return (problems[problems.label == False]
            , problems[problems.label == True])


def get_top_features(v, model, accepted = True, start = 1, end = 10):
    """ Get the most probable n-grams for a given class.

    >>> v = vectorizer(df.body)
    >>> model = create_test_model(v, df)
    >>> get_top_features(v, model)
    """
    i = 1 if accepted else 0
    probs = zip(v.get_feature_names(), model.feature_log_prob_[i])
    return sorted(probs, key = lambda x: -x[1])[start:end]

def pick(i, a):
    return [x[i] for x in a]

class Model(object):
    def __init__(self, models):
        self.models = pick(0, models)
        self.pred_fns = pick(1, models)
        self.transforms = pick(2, models)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X,y)

    def predict_proba(self, X):
        preds = [getattr(m, fn)(X) for m,fn in zip(self.models, self.pred_fns)]
        preds = [t(p) for p,t in zip(preds, self.transforms)]
        return np.array(preds).mean(axis=0)


class OneSidedNB(MultinomialNB):
    """ Uses only the words predictive of the positive class"""
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _joint_log_likelihood(self, X):
        probs = self.feature_log_prob_
        ignore = probs[1,:] < probs[0,:]
        self.feature_log_prob_[:, ignore] = 0
        return super(OneSidedNB, self)._joint_log_likelihood(X)

def get_prediction_data(coll, label, start = datetime(1970,1,1), lookup = None):
    if not lookup:
        lookup = [('ge', 'title'),
                  ('tw', 'body'),
                  ('fa', 'body')]

    # get unique for labelled! Not unique for unlabelled.
    get = lambda s: get_articles(coll, label=label, src=s, date_start=start, unique=label)
    sources = ((get(src),key) for src,key in lookup)
    minimized = (
        {'text': a['content'][key],
         'label': a['label'] == 'accepted' if a.get('label') else None,
         '_id': a['_id']}
        for articles,key in sources for a in articles)
    df = pd.DataFrame(list(minimized))
    if len(df) == 0:
        return [], [], []
    return df.text, df.label, df._id

def base_model():
    models = [(LinearSVC(tol = 10e-6, max_iter = 8000),
               'decision_function',
               lambda n: (n*2.5 + 4)/10),
              (MultinomialNB(class_prior = [0.4,0.6]),
               'predict_proba',
               lambda n: n[:,1])]

    return Model(models)

def train_and_predict(model, X_train, Y_train, X_test):
    idx = X_train.shape[0]
    X = pd.concat([X_train, X_test])
    vector = vectorizer(X).transform(X)
    V_train, V_test = vector[0:idx], vector[idx:]

    model.fit(V_train, Y_train)
    return model.predict_proba(V_test)
