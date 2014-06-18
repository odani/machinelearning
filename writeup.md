Practical Machine Learning project writeup
========================================================

Reading the data into memory and setting blanks to NA

```r
library(caret);
pml <- read.csv("/Volumes/Storage/Dropbox/education/coursera/practical machine learning/project/pml-training.csv", na.strings=c("NA", ""))
```

Partitioning the data into training, validation and test sets. Since we have a large number of observations, we have the luxury of splitting it into three sets like this. This was split 50/25/25. The idea is to use the training set to build the models, then use the validation set in order to get prediction error estimates. Using the prediction error estimates, we can evaluate the models and choose the optimal one. Then the chosen model can be run one last time against the test set, and error rate for the test set can be used to estimate the out-of-sample error rate.

```r
inBuild <- createDataPartition(pml$classe, p = 1/2, list=F)
training <- pml[inBuild,]
build <- pml[-inBuild,]
inBuild2 <- createDataPartition(build$classe, p=1/2, list=F)
validation <- build[inBuild2,]
testing <- build[-inBuild2,]
```

getting rid of the predictors that have NA values. The predictors that had NA values had a large proportion of NAs, and even after removing these predictions, we still have over 50 predictors remaining, so it makes sense to remove them. The alternative would be imputation, which I believe is unnecessary in this case.

```r
NAs <- apply(training,2,function(x) {sum(is.na(x))})
training <- training[,which(NAs == 0)]
```

Getting rid of timestamps, index variable and username since these serve as indexes to the observation, and they shouldn't be used for modeling.

```r
training <- training[, -(1:5)]
```

Finding near zero variance predictors in order to exclude. This shows that I can remove new_window

```r
nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv
rm(nsv)
training <- training[, -1]
```

PCA was not used because there was no real need to reduce the number of dimensions.
Since this is a classification problem, I decided to try CART, Random Forests and Boosted Trees.

The first one up is CART. Model fitting is done as usual on the training set. This algorithm uses bootstrap resampling on the training data to determine the optimal tuning parameter. In this case, it was determined that the optimal tuning parameter was cp=0.392.

```r
fitTree <- train(training$classe ~ ., method="rpart", data=training)
```

The model was then applied to the validation set, and a confusion matrix outputted in order to get the accuracy, which was 0.4868.

```r
predTree <- predict(fitTree, validation)
confusionMatrix(predTree, validation$classe)
```

The next model up is Random Forests. This one uses 10-fold cross validation to tune the model. Bootstrap resampling can also be done, but this is computationally more intensive. The optimal tuning parameter was mtry = 27

```r
fitRf <- train(training$classe ~., data=training, method="rf", trControl = trainControl(method="cv"), number=3)
```

The model was then applied to the validation set, and a confusion matrix outputted in order to get the accuracy, which was 0.9963. It looks like we may have our winner!

```r
predRf <- predict(fitRf, validation)
confusionMatrix(predRf, validation$classe)
```

It's probably not too likely that we will be able to get a better accuracy than we did with Random Forests, but we'll still give Boosted Trees a try. This model uses bootstrap resampling to get the final tuning parameters. In this case, the optimal tuning parameters were found to be n.trees = 150, interaction.depth = 3 and shrinkage = 0.1

```r
modelBoostedTree <- train(training$classe ~., data=training, method="gbm", verbose=F)
```

The model was then applied to the validation set, and a confusion matrix outputted in order to get the accuracy, which was 0.9839. Not bad, but still not as high as Random Forests.

```r
predBoostedTrees <- predict(modelBoostedTree, validation)
confusionMatrix(predBoostedTrees, validation$classe)
```


Given that the winning model is random forest we now want to estimate out of sample error rate by using the test set. The accuracy when applied to the test data set was 0.9963, the same as what we got for the validation set, which I suppose is not surprising.

```r
predRfTest <- predict(fitRf, testing)
confusionMatrix(predRfTest, testing$classe)
```

And finally to get the predictions for the separate 20 test cases...

```r
predRfTestCases <- predict(fitRf, pml.testing)
```
