from mock import patch, MagicMock
from pymongo import UpdateOne, MongoClient as MC
from datetime import datetime
import pandas as pd
import numpy as np
import pytest
from .models import *

@pytest.fixture(scope="module")
def collection():
    client = MC()
    collection = client['newsfilter-test'].news
    yield collection
    collection.drop()
    client.close()


def test_get_prediction_data(collection):
    collection.insert_many([
        {'_id': 'tw:abc', 'title': None, 'content': {'body': 'foo goes to a bar'}, 'label': 'accepted', 'added': datetime.utcnow()},
        {'_id': 'tw:cde', 'title': None, 'content': {'body': 'bar walks down the street'}, 'added': datetime.utcnow()},
        {'_id': 'tw:efg', 'title': None, 'content': {'body': 'bar too town'}, 'added': datetime.utcnow()}
    ])

    dat,_,_ = get_prediction_data(collection, label=False, start=datetime(1970,1,1))
    assert len(dat) == 2
    collection.drop()
