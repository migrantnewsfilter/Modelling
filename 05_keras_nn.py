###WORK IN PROGRESS###

##GENERAL PACKAGES
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from scipy import sparse, io

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

################################################################################
###########################DATA SET UP##########################################
################################################################################

news_feeds_df_tfidf = io.mmread('Data/data_tfidf_' + time.strftime("%Y_%m_%d") +".mtx")
news_feeds_df_tfidf = news_feeds_df_tfidf.tocsr()
news_feeds_df_tfidf = news_feeds_df_tfidf.toarray()

news_feeds_df = pd.DataFrame.from_csv('Data/data_df_' + time.strftime("%Y_%m_%d") +".csv", sep='\t', encoding='utf-8')
#news_feeds_df = news_feeds_df.toarray()
list(news_feeds_df.columns.values)

#Recode Output Layer
news_feeds_df['output'] = news_feeds_df['label'] =='accepted'

frame2['eastern'] = frame2.state == 'Ohio'


################################################################################
#######################KERAS NEURAL NETWORK#####################################
################################################################################

batch_size = 32
nb_epoch = 20
in_out_neurons = news_feeds_df_tfidf.shape[1]
dimof_middle = 100

model = Sequential([
    Dense(batch_size, batch_input_shape=(None, in_out_neurons)),
    Activation('relu'),
    Dropout(0.2),
    Dense(batch_size),
    Activation('relu'),
    Dropout(0.2),
    Dense(in_out_neurons),
    Activation('linear'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

history = model.fit(news_feeds_df_tfidf, np.array(news_feeds_df['output']),
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1
)

"""
, validation_data=(X_test, y_test)


model.1 = Sequential([
    Dense(output_dim=32, input_dim=2145),
    Activation("relu"),
    Dense(10),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

model.fit(news_feeds_df_tfidf, news_feeds_df['label'])

model.fit(news_feeds_df_tfidf, news_feeds_df['label'], nb_epoch=5, batch_size=32)

#Feeding batches to model separately
#model.train_on_batch(X_batch, Y_batch)

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)


################################################################################
###############################BABY EXAMPLE#####################################
################################################################################


X_train = np.array([[1,2], [6,5], [8,2]])
y_train = np.array([2,3,7])
y_train = y_train.reshape((-1, 1))
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(output_dim=64, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=5, batch_size=32)

X_train = np.array(news_feeds_df_tfidf)
y_train = np.array(news_feeds_df['label'])
y_train = y_train.reshape((-1, 1))
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(output_dim=64, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=5, batch_size=32)

df = news_feeds_df_tfidf
numpyMatrix = df.as_matrix()

"""
