from functools import partial
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, VectorizerMixin
from sklearn.cluster import DBSCAN
from modelling.utils import get_articles, clean_html, get_bodies, md5, clean_twitter
from sklearn.feature_extraction.text import strip_accents_unicode
import pandas as pd
import numpy as np
import re
import difflib
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
        'published': d.get('published', d.get('added')),
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

def tokenizer(sent, pattern):
    unigrams = pattern.findall(sent)
    return unigrams

def clustering_preprocessor(s):
    s = clean_html(s)
    s = clean_twitter(s)
    s = strip_accents_unicode(s.lower())
    s = s.strip()
    return s

def compare(a,b,p):
    a,b = tokenizer(a,p), tokenizer(b,p)
    la, lb = len(a), len(b)
    max_len = max(lb, la)
    max_dist = 1/(.1)
    if max_len < 1:
        return max_dist
    s = difflib.SequenceMatcher(None, a, b)
    size = s.find_longest_match(0, la, 0, lb).size
    ratio = size / max_len
    if size > 6 or ratio > 5/6: # I want at least 6 matching words
        return 1/ratio - 1
    return max_dist

def simple_cluster(X, thresh, metric, **kwds):
    N = X.shape[0]
    cls = np.repeat(-1, N)
    out = np.repeat(-1., N*N).reshape((N,N))
    for i in np.arange(N):
        if cls[i] != -1:
            continue
        for j in np.arange(i, N):
            if cls[j] != -1:
                continue
            out[i,j] = metric(X[i], X[j], **kwds)
        new_dists = out[i, i:]
        matches = (new_dists < thresh) & (new_dists > -1)
        cls[i:][np.where(matches)] = i
    return cls

def cluster_articles(data, thresh, body = 'body'):
    """ takes generator of data and returns clusters as numpy array"""

    data = [d for d in data]
    bodies = [get_bodies(d, body) for d in data if d]
    if len(bodies) == 0:
        return []
    p = re.compile(r"(?u)\b\w\w+\b")
    bodies = [clustering_preprocessor(a) for a in bodies]
    cluster_nums = simple_cluster(np.array(bodies), thresh, compare, p=p)
    return hash_cluster(data, cluster_nums).hash_cluster.as_matrix()


def get_cluster_table(thresh, body = 'body', src = 'tw', date = datetime(2017,9,26), db = None):
    """ Used for optimizing and testing interactively """

    collection = MongoClient('209.177.92.45:80')['newsfilter'].news
    data = get_articles(collection, src=src, date_start = date)
    data = [d for d in data]
    fit = cluster_articles(data, thresh)
    bodies = [get_bodies(d, body) for d in data]
    zipped = zip(fit, bodies)
    z = sorted(zipped, key = lambda a: a[0])
    z.reverse()
    return pd.DataFrame(z, columns = ['cluster', 'body'])
