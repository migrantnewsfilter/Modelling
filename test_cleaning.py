from accuracy import *

shipwreck = u'Over 200 <b>migrants die</b> in shipwrecks off Libya. Over 200 <b>migrants die</b> in shipwrecks off Libya. Report Abuse Click Here To Read More. Similar News&nbsp;...'

def test_cleaning ():
    assert 'nbsp' not in clean_html(shipwreck)
