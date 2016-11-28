#!/usr/bin/env python

from pymongo import MongoClient
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN

def get_articles_by_source(src = '', label = ''):
    client = MongoClient()
    collection = client['newsfilter'].news #news - is new one
    cursor = collection.find({ '_id': {'$regex': src }, 'label': {'$regex': label}})
    return list(cursor)

def create_df(li):
    original = pd.DataFrame(li)
    return (pd
     .concat([original, pd.DataFrame(original['content'].tolist())], 1)
     .drop('content', 1)
    )

def get_all_articles():
    real = get_articles_by_source('ge')
    fake = get_articles_by_source('fa')
    return create_df(real + fake)

def get_all_tweets():
    tweets = get_articles_by_source('tw')
    return create_df(tweets)

def dbscan(data, epsilon = 3, samples = 2):
    """ test our dbscan

    - run mongo locally (newsfilter-api: 'docker-compose up')
    - seed database if not seeded

    articles = get_all_articles()
    dbscan(articles.body, 2)
    """

    cleaned = data.values.astype('U')

    c = CountVectorizer(
        stop_words = 'english',
        ngram_range = (2,3)
    ).fit_transform(cleaned)

    db = DBSCAN(
        eps = epsilon,
        min_samples = samples
    ).fit_predict(c)

    for i in set(db):
        print i
        print len(data[ db == i ])
        print data[db == i]
        print ' '

    return db
