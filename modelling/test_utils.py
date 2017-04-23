from utils import clean_html, get_article
from mongomock import MongoClient

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



#########################################################
# get_article
#########################################################

def test_get_article_with_regex():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'foo': 'bar'}, {'_id': 'ge:dbc', 'foo': 'bar'}])
    assert len(get_article(collection, src = 'tw')) == 1
    assert len(get_article(collection, src = 'ge')) == 1
    assert len(get_article(collection)) == 2


def test_get_article_with_label():
    collection = MongoClient().db.collection
    collection.insert_many([{ '_id': 'tw:abc', 'label': 'shite'}, {'_id': 'ge:dbc'}, {'_id': 'ge:boo'}])
    assert len(get_article(collection, False)) == 2
    assert len(get_article(collection, True)) == 1
    assert len(get_article(collection)) == 3
