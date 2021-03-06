from pymongo import MongoClient
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from modelling.utils import get_articles

def get_labelled_articles(uri = None):
    collection = MongoClient(uri)['newsfilter'].news
    return get_articles(collection, label = True)

def create_df(li):
    original = pd.DataFrame(li)
    return (pd
     .concat([original, pd.DataFrame(original['content'].tolist())], 1)
     .drop('content', 1)
    )
