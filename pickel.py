import cPickle
import datetime as dt
from modelling.models import create_model

def pickle(name, model):
    path = 'serialized_classifiers/' + name + '_' + dt.datetime.utcnow().isoformat() + '.pkl'
    with open(path, 'wb') as fout:
        cPickle.dump(model, fout)
