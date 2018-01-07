from functools import partial
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from modelling.utils import get_articles, clean_html
import pandas as pd
import numpy as np
from datetime import datetime

def vectorize(data):
    return  CountVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        preprocessor = clean_html
    ).fit_transform(data)

def dbscan(data, db):
    return db.fit_predict(vectorize(data))

def get_unique_items(df, epsilon, body = 'body'):
    db = DBSCAN(eps = epsilon, min_samples = 2)
    clusters = dbscan(df[body], db)
    df = df.assign(cluster = clusters)
    df_clusters = (df[df.cluster != -1].groupby('cluster')
                   .first())
    return pd.concat([df_clusters, df[df.cluster == -1]])


def get_cluster_table(eps, body = 'body', src = 'tw', date = datetime(2017,9,26), db = None):
    """ Used for optimizing and testing interactively """
    db = DBSCAN(eps = eps, min_samples = 2, algorithm = 'brute', metric = 'cosine') if not db else db
    collection = MongoClient()['newsfilter'].news
    tweets = get_articles(collection, src=src, date_start = date)
    bodies = list(map(lambda x: x['content'].get(body), tweets))
    fit = dbscan(bodies, db)
    zipped = zip(fit, bodies)
    z = sorted(zipped, key = lambda a: a[0])
    z.reverse()
    return pd.DataFrame(z, columns = ['cluster', 'body'])
