import json
from modelling.clustering import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modelling.utils import md5

with open('resources/tweets_a.json') as f:
    tweets_a = json.load(f)

# def test_get_unique_items():
#     df = pd.DataFrame({ 'body': tweets_a, 'label': ['accept'] * len(tweets_a)})
#     uniques = get_unique_items(df, .5)
#     assert len(uniques) == 7
#     assert len(uniques) < len(tweets_a)

tweets = ['RT @IbrahRazan: Nine bodies of #Syrians frozen to death while trying to cross into #Lebanon were found close to the border. Two innocent ch…',
          'Fifteen Syrian refugees - some of them children - have been found frozen to death while trying to cross the mountai… https://t.co/AYU5lAlNhV',
          '#Syrian #refugees were found frozen to death while trying to cross the #border into #Lebanon. https://t.co/h6IttdshOq',
          '15 Syrian refugees - some of them children - have been found frozen to death while trying to cross the mountainous… https://t.co/jyOb9tsN0K'
]


p = re.compile(r"(?u)\b\w\w+\b")
def test_compare_returns_high_when_different():
    a,b,c,d = tweets
    assert compare(b,d,p) < .3
    assert compare(a,a,p) == 0.
    assert compare(a,b,p) > 2.
    assert compare(a,d,p) > 2.

def test_compare_works_with_small_tweets():
    a,b,c = ['foo bar', 'bar foo', 'foo bar']
    assert compare(a,c,p) == 0.
    assert compare(a,b,p) > 5.

def test_compare_works_nothings():
    assert compare('','',p) == 10.

def test_simple_cluster():
    a = np.array(tweets)
    assert simple_cluster(a, 1.0, compare, p=p).tolist() == [0,1,2,1]

def test_get_cluster_articles_with_malformed_data():
    assert cluster_articles([{'foo': 'bar'}], 0.5) == []

def test_cluster_articles_works_with_empty_data():
    assert cluster_articles([], 0.5) == []
    assert cluster_articles(np.array([]), 0.5) == []
    assert cluster_articles(pd.Series([]), 0.5) == []

def test_hash_cluster_adds_hash_of_earliest_in_cluster():
    raw = [('a', datetime(2017, 10, 1), 'foo'),
           ('b', datetime(2017, 10, 15), 'bar'),
           ('c', datetime(2017, 9, 15), 'baz'),
           ('d', datetime(2017, 9, 1), 'bar foo')]

    clusters = [0, -1, -1, 0]
    data = ({'_id': a, 'published': b, 'content': {'body': c}}
            for a,b,c in raw)

    df = hash_cluster(data, clusters)
    assert df.hash_cluster.iloc[0] == md5('bar foo')
    assert df.hash_cluster.iloc[3] == md5('bar foo')
    assert df.body.iloc[0] == 'foo'
    assert df.body.iloc[3] == 'bar foo'

def test_hash_cluster_adds_hash_from_body_variable():
    raw = [('a', datetime(2017, 10, 1), 'foo'),
           ('b', datetime(2017, 9, 15), 'bar')]

    clusters = [0, 0]
    data = ({'_id': a, 'published': b, 'content': {'title': c}}
            for a,b,c in raw)

    df = hash_cluster(data, clusters, 'title')
    assert df.hash_cluster.iloc[0] == md5('bar')
    assert df.body.iloc[0] == 'foo'
    assert df.body.iloc[1] == 'bar'
