#!/usr/bin/env python
execfile("text_processing.py")
##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle

from os import path
from PIL import Image

from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob
from stop_words import get_stop_words

news_feeds_df = pd.DataFrame.from_csv('Data/data_df_' + time.strftime("%Y_%m_%d") +".csv", sep='\t', encoding='utf-8')
#news_feeds_df = pd.DataFrame.from_csv('/Users/robertlange/Desktop/news_filter_project/Modelling/Data/data_df_2016_11_13.csv', sep='\t', encoding='utf-8')

NF_df_tokens = news_feeds_df.text.apply(split_into_tokens)
NF_df_lemmas = news_feeds_df.text.apply(split_into_lemmas)
NF_df_lemmas_stop = NF_df_lemmas.apply(remove_stop_words)
NF_df_lemmas_stop = NF_df_lemmas_stop.values
type(NF_df_lemmas_stop)
text = str(NF_df_lemmas_stop)
isinstance(text, basestring)

text = text.replace("u'", "")
text = text.replace("'", "")
text = text.replace("nbsp", "")
text = text.replace("s", "")

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud)
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# PIL instead of matplotlib
#image = wordcloud.to_image()
#image.show()

world_mask = np.array(Image.open("/Users/robertlange/Desktop/news_filter_project/Modelling/Graphical_Analysis/mask_world.png"))
refugee_mask = np.array(Image.open("/Users/robertlange/Desktop/news_filter_project/Modelling/Graphical_Analysis/mask_refugees.png"))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc_world = WordCloud(background_color="white", max_words=2000, mask=world_mask,
               stopwords=stopwords)
wc_world.generate(text)

wc_refugee = WordCloud(background_color="white", max_words=2000, mask=refugee_mask,
               stopwords=stopwords)
wc_refugee.generate(text)

# store to file
wc_world.to_file("/Users/robertlange/Desktop/news_filter_project/Modelling/Graphical_Analysis/world.png")
wc_refugee.to_file("/Users/robertlange/Desktop/news_filter_project/Modelling/Graphical_Analysis/refugee.png")

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(freedom_mask, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
