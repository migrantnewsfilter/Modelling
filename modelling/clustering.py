from functools import partial
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from modelling.models import clean_html
import pandas as pd
import numpy as np

def dbscan(data, epsilon, n = 2):
    vector = CountVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        preprocessor = clean_html
    ).fit_transform(data)

    db = DBSCAN(eps=epsilon, min_samples=n, algorithm='brute', metric='cosine')
    fit = db.fit_predict(vector)
    return fit


def get_unique_items(df, epsilon, body = 'body'):
    clusters = dbscan(df[body], epsilon)
    df = df.assign(cluster =  clusters)
    df_clusters = (df[df.cluster != -1].groupby('cluster')
                   .first())
    return pd.concat([df_clusters, df[df.cluster == -1]])


def get_cluster_table(epsilon):
    """ Used for optimizing and testing interactively """
    collection = MongoClient()['newsfilter'].news
    tweets = get_all_articles(collection)
    bodies = map(lambda x: x['content'].get('body'), tweets)
    fit = dbscan(bodies, epsilon)
    zipped = zip(fit, bodies)
    z = sorted(zipped, key = lambda a: a[0])
    z.reverse()
    return pd.DataFrame(z, columns = ['cluster', 'body'])
