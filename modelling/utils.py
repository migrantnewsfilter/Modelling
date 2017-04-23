from bs4 import BeautifulSoup
from re import sub

def get_article(collection, label = None, src = ''):
    pattern = { '_id': {'$regex': src }}
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
