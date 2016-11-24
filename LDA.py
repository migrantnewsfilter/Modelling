###WORK IN PROGRESS###

##Unsupervised learning using Latent Dirichlet Allocation Model (Blei, et. al. (2003))

import gensim
from gensim import corpora, models
from scipy import sparse, io
from nltk.tokenize import RegexpTokenizer

##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from text_processing import *



news_feeds_df = pd.DataFrame.from_csv('Data/data_df_' + time.strftime("%Y_%m_%d") +".csv", sep='\t', encoding='utf-8')
list(news_feeds_df.columns.values)

NF_df_tokens = news_feeds_df.text.apply(split_into_tokens)
NF_df_lemmas = news_feeds_df.text.apply(split_into_lemmas)
NF_df_lemmas_stop = NF_df_lemmas.apply(remove_stop_words)
#Problem with stemming - transforms migration to migrat - Do we want this?
NF_df_lemmas_stop_stem = NF_df_lemmas_stop.apply(stemming_words)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(NF_df_lemmas_stop_stem)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=2, num_words=10))
