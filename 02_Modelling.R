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

X.train <- as.data.frame(cbind(data.train$label, data.train$key.word.title, data.train$key.word.body))
X.test <- as.data.frame(cbind(data.test$label, data.test$key.word.title, data.test$key.word.body))

#######################################################################################
#LOGISTIC REGRESSION
#######################################################################################

logistic <- glm(data.train$label ~ ., data=X.train, family='binomial')
log_reg_pred <- predict(logistic,X.test)


#######################################################################################
#NAIVE BAYES
#######################################################################################

library(e1071)
naive.Bayes <-naiveBayes(data.train$label ~ ., data = X.train)
predicted= predict(naive.Bayes,X.test)


#######################################################################################
#DECISION TREES
#######################################################################################

library(rpart)
dec_tree <- rpart(data.train$label ~ ., data = X.train, method="class")
predicted.dec_tree <-  predict(dec_tree, X.test)


#######################################################################################
#SVM
#######################################################################################

library(e1071)
SVM <-svm(data.train$label ~ ., data = X.train)
predicted.svm <- predict(SVM,X.test)
