from modelling.utils import *
from mongomock import MongoClient
from pymongo import MongoClient as MC
from datetime import datetime, timedelta
import pytest

@pytest.fixture(scope="module")
def collection():
    client = MC()
    collection = client['newsfilter-test'].news
    yield collection
    collection.drop()
    client.close()

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
    assert preprocessor(s) == "<SHORT> foo"

def test_preprocessor_removes_commas():
    s = "foo 30,000 baz"
    assert preprocessor(s) == "<SHORT> foo 10000 baz"

def test_split_numbers():
    s = "5yo"
    assert "5yo" not in split_numbers(s)

def test_tokenize_numbers():
    assert tokenize_numbers("foo 5 bar") == "foo 1 bar"
    assert tokenize_numbers("foo 50 bar") == "foo 10 bar"
    assert tokenize_numbers("foo 500 bar") == "foo 100 bar"

def test_tokenize_short():
    assert tokenize_short("foo bar baz") == "<SHORT> foo bar baz"
    assert tokenize_short("foo bar baz"*10) == "foo bar baz"*10
    assert preprocessor("foo bar baz") == "<SHORT> foo bar baz"


#########################################################
# get_articles
#########################################################

def test_get_articles_with_regex():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'added': datetime.utcnow(), 'foo': 'bar'}, {'_id': 'ge:dbc', 'added': datetime.utcnow(), 'foo': 'bar'}])
    assert len(list(get_articles(collection, src = 'tw'))) == 1
    assert len(list(get_articles(collection, src = 'ge')))== 1
    assert len(list(get_articles(collection))) == 2


def test_get_articles_with_label():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite', 'added': datetime.utcnow()},
                            {'_id': 'ge:dbc', 'added': datetime.utcnow()},
                            {'_id': 'ge:boo', 'added': datetime.utcnow()}])
    assert len(list(get_articles(collection, False))) == 2
    assert len(list(get_articles(collection, True))) == 1
    assert len(list(get_articles(collection))) == 3

def test_get_articles_default_start():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite', 'added': datetime.utcnow()},
                            {'_id': 'ge:boo', 'added': datetime(1971, 1, 1)}])
    assert len(list(get_articles(collection))) == 2

def test_get_articles_later_start():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite', 'added': datetime.utcnow()},
                            {'_id': 'ge:boo', 'added': datetime(1971, 1, 1)}])
    assert len(list(get_articles(collection, date_start= datetime.utcnow()))) == 0

def test_get_articles_up_to_is_strictly_greater():
    date_from = datetime.now() - timedelta(weeks = 1)
    old_item = datetime.now() - timedelta(weeks = 2)

    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite', 'added': datetime.utcnow()},
                            {'_id': 'ge:boo', 'added': old_item}])

    assert len(list(get_articles(collection, date_start = old_item))) == 2
    assert len(list(get_articles(collection, date_end = old_item))) == 0
    assert len(list(get_articles(collection, date_start = old_item, date_end = date_from))) == 1


def test_get_articles_unique_gets_timestamp_first(collection):
    date_from = datetime.now() - timedelta(weeks = 7)
    old = datetime.now() - timedelta(weeks = 2)
    older = datetime.now() - timedelta(weeks = 4)
    oldest = datetime.now() - timedelta(weeks = 6)

    collection.insert_many([{ '_id': 'tw:a', 'cluster': 'foo', 'added': datetime.utcnow()},
                            {'_id': 'tw:b', 'cluster': 'bar', 'added': old},
                            {'_id': 'tw:c', 'cluster': 'foo', 'added': older},
                            {'_id': 'tw:d', 'added': oldest}])

    res = list(get_articles(collection, date_start = date_from, unique=True))
    collection.drop()

    assert [d['cluster'] for d in res if d.get('cluster')] == ['bar', 'foo']
    foos = [d for d in res if d.get('cluster') == 'foo']
    assert len(foos) == 1
    assert foos[0]['_id'] == 'tw:c'


def test_get_articles_works_with_unique_and_label(collection):
    date_from = datetime.now() - timedelta(weeks = 7)
    old = datetime.now() - timedelta(weeks = 2)
    older = datetime.now() - timedelta(weeks = 4)
    oldest = datetime.now() - timedelta(weeks = 6)

    collection.insert_many([
        { '_id': 'tw:a', 'cluster': 'foo', 'label': 'accepted', 'added': datetime.utcnow()},
        {'_id': 'tw:b', 'cluster': 'foo', 'label': 'accepted', 'added': old},
        {'_id': 'tw:c', 'cluster': 'bar', 'added': older},
        {'_id': 'tw:d', 'added': oldest}
    ])

    res = list(get_articles(collection, label=True, date_start = date_from, unique=True))
    collection.drop()
    assert [d['cluster'] for d in res if d.get('cluster')] == ['foo']




#########################################################
# md5
#########################################################


def test_md5():
    assert type(md5('foo bar baz')) == str
