from models import clean_html

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
