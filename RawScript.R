library(dplyr)
library(caret)
library(caretEnsemble)
library(e1071)
library(dplyr)
library(tidyr)
library(ROCR)
library(doParallel)
library(c50)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

training = read.csv('./Data/pml-training.csv', na.strings=c("", "NA"))
testing = read.csv('./Data/pml-testing.csv', na.strings=c("", "NA"))

completeCols = which(colSums(is.na(training))==0&names(training)!='X')
training = training[,completeCols]
timeCols = grep('time|window', names(training), value = T)

usernames = unique(training$user_name)
classeUnique = unique(training$classe)
training = select(training, -all_of(timeCols))%>%
    mutate(user_name = factor(user_name, levels = usernames),
           classe = factor(classe, levels = classeUnique))

set.seed(123)
testInd = createDataPartition(training$classe, p = 0.2, list = F)
test = training[testInd,]
train = training[-testInd,]
validationInd = createDataPartition(train$classe, p = 0.25, list = F)
valid = train[validationInd,]
trainE = train[-validationInd,]
rm(train)

gathered = gather(trainE, -classe, -user_name,key = 'var', value = 'value')
# ggplot(gathered, aes(value, classe, colour = user_name))+
#     geom_point()+
#     facet_wrap(~var, scales = 'free')+
#     theme_bw()
windows(12,7,12)   
ggplot(gathered, aes(classe, value, fill = classe, colour = classe))+
    geom_boxplot()+
    facet_wrap(~var, scales = 'free')+
    theme_bw()

trnCntrl = trainControl(method = "cv", number = 5, allowParallel = T, classProbs = T)
# Random forest
set.seed(123)
mod.rf = train(classe~., trainE, method = 'rf', trControl = trnCntrl,
                preProcess = c('center', 'scale', 'pca'), thresh = 0.8)
# SVM with Radial kernal
set.seed(123)
mod.svm = train(classe~., trainE, method = 'svmRadial', trControl = trnCntrl,
                preProcess = c('scale', 'center','pca'), thresh = 0.8)
# C5.0
set.seed(123)
mod.cart <- train(classe ~ ., trainE, method="C5.0", trControl = trnCntrl,
                  preProcess=c("pca","center","scale"), thresh = 0.8)
# K nearest neighbours
set.seed(123)
mod.knn = train(classe~., trainE, method = 'knn', trControl = trnCntrl,
                preProcess=c("pca","center","scale"), thresh = 0.8)
singleRes = list(rf = mod.rf, svm = mod.svm, cart = mod.cart, knn = mod.knn)
summary(resamples(singleRes))$statistics$Accuracy

confusionMatrix(predict(mod.svm.gbm, test), test$classe)

qplot(gyros_dumbbell_x, gyros_dumbbell_y, col = classe, data = trainE)

stopCluster(cluster)
registerDoSEQ()

sum(predict(mod.rf)==predict(mod.svm.pca))

# Voting
createProbDf = function(objList, newdata){
    rf = predict(objList$rf, newdata, type = 'prob')
    names(rf) = paste('rf.', names(rf))
    knn = predict(objList$knn, newdata, type = 'prob')
    names(knn) = paste('knn.', names(knn))
    cart = predict(objList$cart, newdata, type = 'prob')
    names(cart) = paste('cart.', names(cart))
    
    data.frame(rf, knn, cart)
}

ProbDf = createProbDf(singleRes, valid)
ensem = train(ProbDf, valid$classe, method = 'rf', trControl = trnCntrl)

confusionMatrix(predict(ensem, createProbDf(singleRes, test)), test$classe)

finPred = apply(predictions, 1, vote)%>%
    factor(levels = classeUnique)
confusionMatrix(finPred, test$classe)
confusionMatrix(predict(mod.rf, test), test$classe)
singleRes = list(cart = mod.cart, rf = mod.rf, svm = mod.svm,
                           knn = mod.knn, svm.gbm = mod.svm.gbm, svmRad = mod.svm.rad)

a = sapply(singleRes, predict, newdata = train)

testing = testing[,completeCols]%>%
    select(testing, -all_of(timeCols))%>%
    mutate(user_name = factor(user_name, levels = usernames),
           classe = factor(classe, levels = classeUnique))

head(training)

