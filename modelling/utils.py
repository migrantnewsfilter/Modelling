from __future__ import print_function
from bs4 import BeautifulSoup
from re import sub, split, findall
from datetime import datetime
import math
from sklearn.feature_extraction.text import strip_accents_unicode

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
        print(e)
        return ''

def split_numbers(s):
    return ' '.join(split('(\d+)[^\d\s]+', s))

def round_numbers(m):
    n = int(m.group(1))
    if n < 1:
        return ''
    i = 10**math.floor(math.log10(n))
    return str(i)

def tokenize_numbers(s):
    return sub('(\d+)', round_numbers, s)

def format_numbers(s):
    decomma = lambda m: m.group(1) + m.group(2)
    s = sub('(\d+),(\d+)', decomma, s)
    return s

def preprocessor(s):
    s = clean_html(s)
    s = format_numbers(s)
    s = split_numbers(s)
    s = tokenize_numbers(s)
    return strip_accents_unicode(s.lower())
