from __future__ import print_function
from bs4 import BeautifulSoup
import re
from re import sub, split, findall
from datetime import datetime, timedelta
import math, hashlib
from sklearn.feature_extraction.text import strip_accents_unicode
import logging
logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def get_bodies(article, body = 'body'):
    try:
        body =  article['content'][body]
    except KeyError as e:
        # logging.error('Malformed article in DB!: ', e)
        return None
    return body

def get_articles(collection, label = None, src = '',
                 date_start = datetime(1970, 1, 1),
                 date_end = datetime.utcnow() + timedelta(hours = 1),
                 unique = False):
    pattern = { '_id': {'$regex': src }, 'added': {'$gte': date_start, '$lt': date_end}}
    if label != None:
        pattern['label'] = {'$exists': label}
    if unique != False:
        pattern['cluster'] = {'$exists': True}
        agg = collection.aggregate([
            { '$match': pattern},
            { '$sort': { 'added': 1 } },
            { '$group': { '_id': '$cluster', 'item': { '$first': '$$ROOT' }}}
        ])
        return map(lambda x: x['item'], agg)

    return collection.find(pattern).sort('added',1)

def clean_html(s):
    """ Converts all HTML elements to Unicode """
    try:
        s = sub(r'https?://[^\s]+', '', s)
        return BeautifulSoup(s, 'html5lib').get_text() if s else ''
    except UserWarning:
        return ''
    except Exception as e:
        logger.debug(e)
        return ''

def clean_twitter(s):
    """ Cleans Twitter specific issues

    Can you think of what else you might need to add here?
    """
    s = sub(r'@\w+', '', s) #remove @ mentions from tweets
    s = re.sub(r'^rt', '', s, flags = re.I)
    return s

def split_numbers(s):
    return ' '.join(split('(\d+)[^\d\s]+', s))

def round_numbers(m, lim = 300):
    n = int(m.group(1))
    if n < 1:
        return ''
    if n < lim:
        return 'SMALLNUMBER'
    else:
        return 'LARGENUMBER'

def tokenize_numbers(s):
    return sub('(\d+)', round_numbers, s)

def tokenize_short(s, lim = 5):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    tokens = token_pattern.findall(s)
    if (len(tokens) < lim):
        return 'SHORTARTICLE ' + s
    else:
        return s

def format_numbers(s):
    decomma = lambda m: m.group(1) + m.group(2)
    s = sub('(\d+),(\d+)', decomma, s)
    return s

def preprocessor(s):
    s = clean_html(s)
    s = clean_twitter(s)
    s = format_numbers(s)
    s = split_numbers(s)
    s = tokenize_numbers(s)
    s = strip_accents_unicode(s.lower())
    s = tokenize_short(s)
    return s

def md5(s):
    h = hashlib.new('md5')
    h.update(s.encode('utf-8'))
    return h.hexdigest()
