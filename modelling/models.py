from pymongo import MongoClient
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline

def get_labelled_articles(client):
    collection = client['newsfilter'].news
    cursor = collection.find({ 'label': {'$exists': True}})
    return list(cursor)

def create_df(li):
    original = pd.DataFrame(li)
    return (pd
     .concat([original, pd.DataFrame(original['content'].tolist())], 1)
     .drop('content', 1)
    )

def vectorizer(df):
    """ Creates a tfidf vectorizer on df.body values """
    return TfidfVectorizer(
        stop_words = 'english',
        ngram_range = (2,4),
        preprocessor = clean_html
    ).fit(df.body.values.astype('U'))

def create_tfidf(v, df):
    return v.transform(df.body.values.astype('U'))

def fit_naive_bayes(tfidf, target, priors):
    """ Shows cross validated score from a NB model with given priors """
    return cross_val_score(MultinomialNB(class_prior=priors), tfidf, target, cv = 50)

def add_predictions(model, df):
    """ Creates a dataframe with predictions from model as 'predicts' """
    c = create_tfidf(vectorizer(df), df)
    predictions = cross_val_predict(model, c, df.label, cv=100)
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


def clean_html(s):
    """ Converts all HTML elements to Unicode """
    try:
        return BeautifulSoup(s, 'html5lib').get_text()
    except:
        return ''

def create_model(data, target, priors = [0.3, 0.7]):
    """ Creates a model/pipeline for production use.

    Returns predict_proba method to be used on an iterable
    of strings!

    data : Iterable of text to be classified
    target: Iterable of target classes (two)
    priors: List (length 2) of prior probabilities of classes
    """
    pipeline = Pipeline([
        ('tfidf',  TfidfVectorizer(stop_words = 'english',
                                   ngram_range = (2,4),
                                   preprocessor = clean_html)),
        ('classifier', MultinomialNB(class_prior=priors))
    ])

    return pipeline.fit(data, target)
