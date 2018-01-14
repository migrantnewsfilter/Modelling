from functools import partial
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from modelling.utils import get_articles, clean_html, get_bodies, md5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def vectorizer(data):
    return  CountVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        preprocessor = clean_html
    ).fit_transform(data)

def dbscan(data, db):
    # transform list into hashes based on the chronological first
    return db.fit_predict(vectorizer(data))

def add_cluster(n, d, body):
    return {
        '_id': d['_id'],
        'published': d['published'],
        'cluster': n,
        'body': d['content'][body]
    }

def apply_hash(group):
    group['hash_cluster'] = md5(group.sort_values('published').body.iloc[0])
    return group

def hash_cluster(data, clusters, body = 'body'):
    df = pd.DataFrame([add_cluster(n,d,body) for n,d
                       in zip(clusters, data)])
    solos = df.cluster == -1
    clustered = df.cluster != -1
    df['hash_cluster'] = None
    df.loc[clustered, :] = (df[clustered]
                            .groupby('cluster', sort=False)
                            .apply(apply_hash))
    df.loc[solos, 'hash_cluster'] = df[solos].body.map(md5)
    return df

def cluster_articles(data, eps, body = 'body'):
    """ takes generator of data and returns clusters as numpy array"""
    if len(data) == 0:
        return []
    data = list(data)
    bodies = map(lambda b: get_bodies(b, body), data)
    db = DBSCAN(eps, min_samples = 2)
    cluster_nums = db.fit_predict(vectorizer(bodies))
    return hash_cluster(data, cluster_nums).hash_cluster.as_matrix()


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
