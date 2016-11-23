#!/usr/bin/env python
from text_processing import *
from data_processing import *

##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle

import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import datetime
#import mpld3

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram, linkage

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

###################################################################

#path = './Data/MMP_all_data.csv'

data = feature_gen()
list(data.columns.values)

pd.to_datetime(data['published'])
data['published']

s = pd.Series(data['published'])
s.reset_index()

pd.to_datetime(s)

tfidf_matrix = tfidf_trafo(data,3)

###################################################################
dist = 1 - cosine_similarity(tfidf_matrix)

#Ward Variance Minimization Algorithm
#Other metrics: 'single', 'complete', 'average'
#Use Ward method to compute distance between newly formed clusters
linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances


fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right"); #, labels=titles

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('./Graphical_Analysis/ward_clusters.png', dpi=200) #save figure as ward_clusters
plt.close()
'''

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


#Cophenetic Correlation Coefficient of clustering
#Compares (correlates) the actual pairwise distances of all your samples to
#those implied by the hierarchical clustering

#c, coph_dists = cophenet(tfidf_matrix , pdist(tfidf_matrix))
#c

##Coming up with the right number of clusters!
'''
#Set numbers of wanted cluseters by max_d
from scipy.cluster.hierarchy import fcluster
max_d = 10
clusters = fcluster(linkage_matrix, max_d, criterion='maxclust')
clusters
clusters.shape

#criterion='maxclust'
