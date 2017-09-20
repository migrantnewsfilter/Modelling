import json
from modelling.clustering import *
import pandas as pd

with open('resources/tweets_a.json') as f:
    tweets_a = json.load(f)

def test_get_unique_items():
    df = pd.DataFrame({ 'body': tweets_a, 'label': ['accept'] * len(tweets_a)})
    uniques = get_unique_items(df, .5)
    assert len(uniques) == 7
    assert len(uniques) < len(tweets_a)
