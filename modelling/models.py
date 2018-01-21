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

def create_df(li):
    li = list(li)
    original = pd.DataFrame(li)
    return (pd
     .concat([original, pd.DataFrame(original['content'].tolist())], 1)
     .drop('content', 1)
    )

def vectorizer(dat, model = TfidfVectorizer):
    """ Creates a tfidf vectorizer on df.body values """
    return model(
        stop_words = 'english',
        ngram_range = (1,3),
        preprocessor = preprocessor
    ).fit(dat)

def create_word_count(df):
    X = df.body.values.astype('U')
    v = vectorizer(X, CountVectorizer)
    return v.transform(X)

def create_tfidf(df):
    X = df.body.values.astype('U')
    v = vectorizer(X)
    return v.transform(X)

def fit_naive_bayes(tfidf, target, priors):
    """ Shows cross validated score from a NB model with given priors """
    logo = LeaveOneGroupOut()
    return cross_val_score(MultinomialNB(class_prior=priors), tfidf, target, cv = 50)

def add_predictions(model, df, cv):
    """ Creates a dataframe with predictions from model as 'predicts' """
    c = create_tfidf(vectorizer(df), df)
    predictions = cross_val_predict(model, c, df.label, cv=cv)
    predicts = pd.DataFrame(predictions, columns = ['predicts'])
    combined = pd.concat([df, predicts], 1)
    problems = combined.loc[combined['label'] != combined['predicts']][['body', 'label', 'predicts']]
    return problems

def create_test_model(v, df, priors = [0.3,0.7]):
    """ Creates a NB Model & fits it via body and label """
    model = MultinomialNB(class_prior = priors)
    model.fit(v.transform(map(clean_html, df.body.values.astype('U'))), df.label)
    return model

def get_top_features(v, model, accepted = True, start = 1, end = 10):
    """ Get the most probable n-grams for a given class.

    >>> v = vectorizer(df.body)
    >>> model = create_test_model(v, df)
    >>> get_top_features(v, model)
    """
    i = 0 if accepted else 1
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


def get_prediction_data(coll, label, start = datetime(1970,1,1)):
    lookup = [('ge', 'title'),
              ('tw', 'body'),
              ('fa', 'body')]

    get = lambda s: get_articles(coll, label=label, src=s, date_start=start, unique=True)
    sources = ((get(src),key) for src,key in lookup)
    minimized = (
        {'text': a['content'][key],
         'label': a['label'] == 'accepted' if a.get('label') else None,
         '_id': a['_id']}
        for articles,key in sources for a in articles)
    df = pd.DataFrame(list(minimized))
    return df.text, df.label, df._id


def train_and_predict(X_train, Y_train, X_test):
    idx = X_train.shape[0]
    X = pd.concat([X_train, X_test])
    vector = vectorizer(X).transform(X)
    V_train, V_test = vector[0:idx], vector[idx:]

    models = [(LinearSVC(tol = 10e-6, max_iter = 8000),
               'decision_function',
               lambda n: (n*2.5 + 4)/10),
              (MultinomialNB(class_prior = [0.4,0.6]),
               'predict_proba',
               lambda n: n[:,1])]

    model = Model(models)
    model.fit(V_train, Y_train)
    return model.predict_proba(V_test)
