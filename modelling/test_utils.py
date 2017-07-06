from modelling.utils import *
from mongomock import MongoClient
from datetime import datetime
#########################################################
# clean_html
#########################################################

shipwreck = u'Over 200 <b>migrants die</b> in shipwrecks off Libya. Over 200 <b>migrants die</b> in shipwrecks off Libya. Report Abuse Click Here To Read More. Similar News&nbsp;...'

retweet = "RT @AmericanPresRS: 'Somali refugee' dead after going on rampage at Ohio State University https://t.co/wN0a3HAE18"

def test_cleaning ():
    assert 'nbsp' not in clean_html(shipwreck)

def test_link_removal():
    cleaned = clean_html(retweet)
    assert 'https://t.co/wN0a3HAE18' not in cleaned
    assert 'Ohio State University' in cleaned

def test_handle_removal():
    cleaned = clean_html(retweet)
    assert '@AmericanPresRs' not in cleaned
    assert 'RT' in cleaned
    assert 'Somali refugee' in cleaned

def test_preprocessor_lowercases_and_accents():
    s = "Foó"
    assert preprocessor(s) == "foo"

def test_preprocessor_removes_commas():
    s = "foo 30,000 baz"
    assert preprocessor(s) == "foo 10000 baz"

def test_split_numbers():
    s = "5yo"
    assert "5yo" not in split_numbers(s)

def test_tokenize_numbers():
    assert tokenize_numbers("foo 5 bar") == "foo 1 bar"
    assert tokenize_numbers("foo 50 bar") == "foo 10 bar"
    assert tokenize_numbers("foo 500 bar") == "foo 100 bar"


#########################################################
# get_articles
#########################################################

def test_get_articles_with_regex():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'added': datetime.utcnow(), 'foo': 'bar'}, {'_id': 'ge:dbc', 'added': datetime.utcnow(), 'foo': 'bar'}])
    assert len(get_articles(collection, src = 'tw')) == 1
    assert len(get_articles(collection, src = 'ge')) == 1
    assert len(get_articles(collection)) == 2


def test_get_articles_with_label():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite', 'added': datetime.utcnow()}, {'_id': 'ge:dbc', 'added': datetime.utcnow()}, {'_id': 'ge:boo', 'added': datetime.utcnow()}])
    assert len(get_articles(collection, False)) == 2
    assert len(get_articles(collection, True)) == 1
    assert len(get_articles(collection)) == 3

def test_get_articles_default_start():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite', 'added': datetime.utcnow()}, {'_id': 'ge:boo', 'added': datetime(1971, 1, 1)}])
    assert len(get_articles(collection)) == 2

def test_get_articles_later_start():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite', 'added': datetime.utcnow()}, {'_id': 'ge:boo', 'added': datetime(1971, 1, 1)}])
    assert len(get_articles(collection, date_start= datetime.utcnow())) == 0
