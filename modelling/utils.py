from bs4 import BeautifulSoup
from re import sub
from datetime import datetime

def get_articles(collection, label = None, src = '', date_start = datetime(1970, 1, 1)):
    pattern = { '_id': {'$regex': src }, 'added': {'$gte': date_start}}
    if label != None:
        pattern['label'] = {'$exists': label}
    return list(collection.find(pattern))

def clean_html(s):
    """ Converts all HTML elements to Unicode """
    try:
        s = sub(r'https?://[^\s]+', '', s)
        s = sub(r'@\w+', '', s)
        return BeautifulSoup(s, 'html5lib').get_text() if s else ''
    except UserWarning:
        return ''
    except Exception as e:
        print e
        return ''
