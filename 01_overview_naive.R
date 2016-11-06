##Author: Robert Lange
##DATE: 05/11/2016
##BGSE - ComputingLab & DataWarehousing Project
#############################################################
#############################################################
##############Text Mining - News Classification##############
#############################################################
#############################################################

rm(list=ls())
cat("\014")
Sys.setenv(LANG = "en")

#############################################################
#PACKAGE SETUP
#############################################################

install.packages("gtools", dependencies = T)
library(gtools) # if problems calling library, install.packages("gtools", dependencies = T)
library(qdap) # qualitative data analysis package (it masks %>%)
library(tm) # framework for text mining; it loads NLP package
library(Rgraphviz) # depict the terms within the tm package framework
library(SnowballC); library(RWeka); library(rJava); library(RWekajars)  # wordStem is masked from SnowballC
library(chron) #time conversion
#library(Rstem) # stemming terms as a link from R to Snowball C stemmer
packages <- c("RColorBrewer", "ggplot2", "wordcloud", "biclust", "cluster", "igraph", "fpc")
install.packages(packages, dependencies=TRUE)
install.packages("Rcampdf", repos = "http://datacube.wu.ac.at/", type = "source")
library(ggplot2)
library(wordcloud)

#############################################################
#LOAD IN DATA - SPLIT TRAIN/TEST - CREATE RANDOM 0/1 LABEL
#############################################################

setwd("/Users/robertlange/Desktop/news_filter_project/random")
load_data <- "alerts_6_11_b.csv"
data <- read.csv(load_data, header=TRUE, , sep=",")
data <- as.data.frame(data)
head(data); dim(data); summary(data)

#############################################################

date_time <- as.character(published)
date <- sapply(strsplit(date_time, split=' ', fixed=TRUE), `[`, 1)
date <- strptime(date, "%Y-%m-%d")
time <- sapply(strsplit(date_time, split=' ', fixed=TRUE), `[`, 2)
time <- sapply(strsplit(time, split='+', fixed=TRUE), `[`, 1)
time <- chron(times=time)

#############################################################
######################## TEXT MINING ########################
#############################################################
data$content.title <- as.character(data$content.title)
data$content.body <- as.character(data$content.body)

#Treats Title and Summary as n documents
docs <- Corpus(VectorSource(title))

#CLEANING OF TEXT DATA
#Rm punctuation, numbers, convert to lower case
#Rm meaningless words, stemming(cut -ing -es etc.)
#Save data as plain text

key.word.extract <- function(x, string){
    ind <- grep(string,x,value = FALSE)
    x <- gsub(string, "\\1", x)
    x[-ind] <- ""
    x
}

data$label <- as.factor(data$label)
data$key.word.title <- as.factor(key.word.extract(data$content.title, ".*?<b>(.*?)</b>.*"))
data$key.word.body <- as.factor(key.word.extract(data$content.body, ".*?<b>(.*?)</b>.*"))
docs <- tm_map(docs, content_transformer(gsub), pattern = '<b>', replacement = '')
docs <- tm_map(docs, content_transformer(gsub), pattern = '</b>', replacement = '')
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, tolower)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removeWords, stopwords("spanish"))
#docs <- tm_map(docs, removeWords, c("department", "email"))
docs <- tm_map(docs, stemDocument)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, PlainTextDocument)

#############################################################
#STAGING OF DATA
#############################################################
tdm <- TermDocumentMatrix(docs) # create tdm_title from the corpus
dtm <- DocumentTermMatrix(docs)
dtm
inspect(dtm[1:5, 1:20])

freq <- colSums(as.matrix(dtm))
length(freq)
ord <- order(freq)

freq[head(ord)]
freq[tail(ord)]
#Frequency of Frequencies
head(table(freq), 20)
tail(table(freq), 20)

m <- as.matrix(dtm)
dim(m)
write.csv(m, file="dtm.csv")

# specifying a correlation limit of 0.98
findAssocs(dtm, c("death" , "migrant"), corlimit=0.3)

#############################################################
#ANALYTICAL GRAPHICS - Word Frequencies, Word Clouds
#############################################################

wf <- data.frame(word=names(freq), freq=freq)
p <- ggplot(subset(wf, freq>10), aes(word, freq))
p <- p + geom_bar(stat="identity")
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))
p

wordcloud(names(freq), freq, min.freq=25)
wordcloud(names(freq), freq, max.words=100)
wordcloud(names(freq), freq, max.words=100, scale=c(5, .1), colors=brewer.pal(6, "Dark2"))

dark2 <- brewer.pal(6, "Dark2")
wordcloud(names(freq), freq, max.words=100, rot.per=0.2, colors=dark2)

# Create max 10% empty matrix:

dtms <- removeSparseTerms(dtm, 0.9)
library(cluster)
d <- dist(t(dtms), method="euclidian")
fit <- hclust(d=d, method="ward")
fit
plot(fit, hang=-1)
groups <- cutree(fit, k=5)   # "k=" defines the number of clusters you are using
rect.hclust(fit, k=5, border="red") # draw dendogram with red borders around the 5 clusters


library(fpc)
d <- dist(t(dtmss), method="euclidian")
kfit <- kmeans(d, 2)
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)


#######################################################################################
#UNSUPERVISED LEARNING - Hierarchical Clustering - Based on dissimilarity of documents
#######################################################################################
library(stringi)
library(proxy)

titles <- paste("doc",seq(1:dim(dtms)[1]))
docsdissim <- proxy::dist(as.matrix(dtms), method = "cosine")
docsdissim2 <- as.matrix(docsdissim)
rownames(docsdissim2) <- titles
colnames(docsdissim2) <- titles
docsdissim2
h <- hclust(docsdissim, method = "ward")
plot(h, labels = titles, sub = "")

dend1 <- as.dendrogram(h)

# Get the package:
install.packages("dendextend")
library(dendextend)

# Get the package:
label <- cutree(dend1,k=2)

#############################################################

smp_size <- floor(0.75 * nrow(data))
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

data.train <- data[train_ind, ]
data.test <- data[-train_ind, ]


#######################################################################################
#NAIVE BAYES CLASSIFIER
#######################################################################################

X.train <- as.data.frame(cbind(data.train$label, data.train$key.word.title, data.train$key.word.body))
X.test <- as.data.frame(cbind(data.test$label, data.test$key.word.title, data.test$key.word.body))

logistic <- glm(data.train$label ~ ., data=X.train, family='binomial')
log_reg_pred <- predict(logistic,X.test)
