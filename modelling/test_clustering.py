import json
from modelling.clustering import *
import pandas as pd
from datetime import datetime, timedelta
from modelling.utils import md5

with open('resources/tweets_a.json') as f:
    tweets_a = json.load(f)

# def test_get_unique_items():
#     df = pd.DataFrame({ 'body': tweets_a, 'label': ['accept'] * len(tweets_a)})
#     uniques = get_unique_items(df, .5)
#     assert len(uniques) == 7
#     assert len(uniques) < len(tweets_a)


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
