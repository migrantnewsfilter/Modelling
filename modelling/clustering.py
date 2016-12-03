from functools import partial
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from models import clean_html
import pandas as pd
import numpy as np

def get_articles_by_source(client, src = ''):
    collection = client['newsfilter'].news
    cursor = collection.find({ '_id': {'$regex': src }, 'label': {'$exists': None}})
    return list(cursor)

def get_all_articles(client):
    real = get_articles_by_source(client, 'ge')
    fake = get_articles_by_source(client, 'fa')
    return real + fake

def get_all_tweets(client):
    tweets = get_articles_by_source(client, 'tw')
    return tweets

def dbscan(data, epsilon, n = 2):
    vector = CountVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        preprocessor = clean_html
    ).fit_transform(data)

    db = DBSCAN(eps=epsilon, min_samples=n, algorithm='brute', metric='cosine')
    fit = db.fit_predict(vector)
    return fit

def get_cluster_table(epsilon):
    client = MongoClient()
    tweets = get_all_tweets(client)
    bodies = map(lambda x: x['content'].get('body'), tweets)
    fit = dbscan(bodies, epsilon)
    zipped = zip(fit, bodies)
    z = sorted(zipped, key = lambda a: a[0])
    z.reverse()
    return pd.DataFrame(z, columns = ['cluster', 'body'])
