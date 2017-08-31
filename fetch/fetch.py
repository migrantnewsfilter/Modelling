from pymongo import MongoClient
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN

def get_labelled_articles(uri = None):
    client = MongoClient(uri)
    collection = client['newsfilter'].news
    cursor = collection.find({ 'label': {'$exists': True}})
    return list(cursor)

def create_df(li):
    original = pd.DataFrame(li)
    return (pd
     .concat([original, pd.DataFrame(original['content'].tolist())], 1)
     .drop('content', 1)
    )
