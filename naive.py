%matplotlib inline
import os
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas as pd
import sklearn
import cPickle
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
from pymongo import MongoClient, ASCENDING
from bson.json_util import dumps
################################################################################
############################LOADING IN OF DATA SET##############################
################################################################################
os.chdir('..')
client = MongoClient("mongodb://209.177.92.45:80")

collection = client['newsfilter'].alerts

def get_articles():
    print 'get articles!!!'
    cursor = collection.find().sort('added', ASCENDING)
    return dumps(cursor)

news_feeds_json = get_articles()
parsed = json.loads(news_feeds_json)
print json.dumps(parsed, indent=4, sort_keys=True)

news_feeds = pd.read_json(news_feeds_json)
news_feeds_df = pd.DataFrame(news_feeds)

temp = news_feeds_df['content'].apply(pd.Series)
news_feeds_df = pd.concat([news_feeds_df, temp], axis=1)
news_feeds_df['text'] =  news_feeds_df['title'] + " " + news_feeds_df['body']

################################################################################
##############################OVERVIEW OF DATA SET##############################
################################################################################

#Drop observations that are not labelled
news_feeds_df = news_feeds_df.dropna()
news_feeds_df['label']

print  "Size of data set:", len(news_feeds_df)
news_feeds_df.head(n=10)
news_feeds_df.groupby('label').describe()

#DEFINE LENGHT OF TITLE + CONTENT BODY
news_feeds_df['length'] = news_feeds_df['text'].map(lambda text: len(text))
news_feeds_df.length.plot(bins=20, kind='hist')
plt.savefig('Histogram_Text_Length.png')
news_feeds_df.length.describe()
print list(news_feeds_df.text[news_feeds_df.length > 350])
news_feeds_df.hist(column='length', by='label', bins=50)
plt.savefig('Histogram_Text_Length_Label.png')

################################################################################
#################################DATA PROCESSING################################
################################################################################

def split_into_tokens(text):
    #split a message into its individual words
    #text = unicode(text, 'utf8') - convert bytes into proper unicode - does not work because already unicode
    return TextBlob(text).words
news_feeds_df.text.head().apply(split_into_tokens)

def split_into_lemmas(text):
    #normalize words into their base form (lemmas)
    text = text.lower()
    words = TextBlob(text).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]
news_feeds_df.text.head().apply(split_into_lemmas)

################################################################################
#############################CONVERT TOKENS TO VECTOR###########################
################################################################################

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(news_feeds_df['text'])
#Each vector has as many dimensions as there are unique words in the text corpus
print len(bow_transformer.vocabulary_)

#Example bag-of-words counts as a vector using bow_transformer
feed4 = news_feeds_df['text'][3]
print feed4
bow4 = bow_transformer.transform([feed4])
print bow4
print bow4.shape
print bow_transformer.get_feature_names()[64]
print bow_transformer.get_feature_names()[1893]

news_feeds_df_bow = bow_transformer.transform(news_feeds_df['text'])
print 'sparse matrix shape:', news_feeds_df_bow.shape #dim: number feeds x unique words
print 'number of non-zeros:', news_feeds_df_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * news_feeds_df_bow.nnz / (news_feeds_df_bow.shape[0] * news_feeds_df_bow.shape[1]))

################################################################################
#############################tfidf - transformation#############################
################################################################################

#Term weighting and normalization can be done with TF-IDF
tfidf_transformer = TfidfTransformer().fit(news_feeds_df_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4

print tfidf_transformer.idf_[bow_transformer.vocabulary_['migrant']]
print tfidf_transformer.idf_[bow_transformer.vocabulary_['death']]

#transform the entire bag-of-words corpus into TF-IDF corpus at once:
news_feeds_df_tfidf = tfidf_transformer.transform(news_feeds_df_bow)
print news_feeds_df_tfidf.shape

################################################################################
####################################Modelling###################################
################################################################################

#Naive Bayes using scikit-learn
%time spam_detector = MultinomialNB().fit(news_feeds_df_tfidf, news_feeds_df['label'])

print 'predicted:', spam_detector.predict(tfidf4)[0]
print 'expected:', news_feeds_df.label[3]

all_predictions = spam_detector.predict(news_feeds_df_tfidf)
print all_predictions

print 'accuracy', accuracy_score(news_feeds_df['label'], all_predictions)
print 'confusion matrix\n', confusion_matrix(news_feeds_df['label'], all_predictions)
print '(row=expected, col=predicted)'

plt.matshow(confusion_matrix(news_feeds_df['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.savefig('Confusion_Matrix.png')

print classification_report(news_feeds_df['label'], all_predictions)

msg_train, msg_test, label_train, label_test = \
    train_test_split(news_feeds_df['text'], news_feeds_df['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

scores = cross_val_score(pipeline,  # steps to convert raw news_feeds_df into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores

print scores.mean(), scores.std()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

%time plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

%time nb_detector = grid.fit(msg_train, label_train)
print nb_detector.grid_scores_

print nb_detector.predict_proba(["Hi mom, how are you?"])[0]
print nb_detector.predict_proba(["WINNER! Credit for free!"])[0]

print nb_detector.predict(["Hi mom, how are you?"])[0]
print nb_detector.predict(["WINNER! Credit for free!"])[0]

predictions = nb_detector.predict(msg_test)
print confusion_matrix(label_test, predictions)
print classification_report(label_test, predictions)

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

%time svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
print svm_detector.grid_scores_

print svm_detector.predict(["Hi mom, how are you?"])[0]
print svm_detector.predict(["WINNER! Credit for free!"])[0]

print confusion_matrix(label_test, svm_detector.predict(msg_test))
print classification_report(label_test, svm_detector.predict(msg_test))

# store the spam detector to disk after training
with open('sms_spam_detector.pkl', 'wb') as fout:
    cPickle.dump(svm_detector, fout)

# ...and load it back, whenever needed, possibly on a different machine
svm_detector_reloaded = cPickle.load(open('sms_spam_detector.pkl'))

print 'before:', svm_detector.predict([feed4])[0]
print 'after:', svm_detector_reloaded.predict([feed4])[0]
