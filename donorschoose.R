library(caret)
# donations and outcomes data sets are available for training only
# imbalanced class

# loading datasets
projects <- read.csv("/Volumes/Storage/temp/projects.csv")
outcomes <- read.csv("/Volumes/Storage/temp/outcomes.csv")
essays <- read.csv("/Volumes/Storage/temp/essays.csv")
# donations <- read.csv("/Volumes/Storage/temp/donorschoose/donations.csv")

# process the essays
a <- apply(essays, 1, function(x){nchar(x['essay'])})
a1 <- data.frame(projectid=essays$projectid, essay.length=a)
projects <- merge(projects, a1, by="projectid")
rm(a); rm(a1); rm(essays)

# splitting between the main data and scoring data
# also want to remove data prior to 4/14/2010 since the outcome class variable
# is all negative before then
projects$date_posted <- as.Date(projects$date_posted)
scoring <- projects[projects$date_posted >= as.Date("2014-01-01") ,]
a <- projects$date_posted; b <- a >= "2010-04-14" & a < "2014-01-01"
data <- projects[b,]
rm(a); rm(b)

# add class variable to the data
o <- outcomes[,1:2]
data <- merge(data, o, by="projectid"); rm(o)

# create training data set for downsampling
set.seed(1234)
split <- createDataPartition(y=data$is_exciting, p=.14, list=FALSE)
a <- data[split,]; other <- data[-split,]
training.down.sampled <- downSample(x=a[, -ncol(a)],
                           y=a$is_exciting, yname="is_exciting")
rm(a); rm(split)

# create training data set for SMOTE
library(DMwR)
set.seed(5342)
split2 <- createDataPartition(y=other$is_exciting, p=.06, list=FALSE)
training <- other[split2,]; other2 <- other[-split2,]
training$date_posted <- as.numeric(training$date_posted)
set.seed(1237)
training.smoted <- SMOTE(is_exciting ~ ., data = training)
rm(training); rm(split2); rm(other)

# create training, evaluation and test sets  
split <- createDataPartition(y=other2$is_exciting, p=.04, list=FALSE)
training <- other2[split,]; other3 <- other2[-split,]; rm(other2)
split <- createDataPartition(y=other3$is_exciting, p=.03, list=FALSE)
evaluation <- other3[split,]; other4 <- other3[-split,]; rm(other3)
split <- createDataPartition(y=other4$is_exciting, p=.03, list=FALSE)
test <- other4[split,]; rm(split); rm(other4)

# impute median for missing values for vars students_reached and
# fulfillment_labor_materials
median1 <- median(data$students_reached, na.rm=T)
median2 <- median(data$fulfillment_labor_materials, na.rm=T)
dflist <- list(training.down.sampled, training.smoted, training, evaluation, test, scoring)

impute <- function(x) {
  x$students_reached[is.na(x$students_reached)] <- median1
  x$fulfillment_labor_materials[is.na(x$fulfillment_labor_materials)] <- median2
}
# for some reason, this isn't doing the imputation for scoring
lapply(dflist, impute)
rm("median1", "median2", "dflist", "impute")

# examined the variance of the predictors but the predictors with low variance
# are all factors, so I want to keep those.
# nearZeroVar(training, saveMetrics=TRUE)

# some columns to exclude from training
exclude1 <- c("projectid", "teacher_acctid", "schoolid", "school_ncesid",
              "school_latitude", "school_longitude", "school_city",
              "school_state", "school_zip", "school_district", "school_county",
              "date_posted")

a <- which(names(training) %in% exclude1)

fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
ctrl <- trainControl(method="cv", summaryFunction=fiveStats,
                       classProbs=T)

# got the best AUC from Stochastic Gradient Boosting (down sampled) an RF (down
# sampled)
### Random Forest
set.seed(1410)
# ROC of 0.55 with mtry=27.
library(doMC)
registerDoMC(cores = 4)
rfFit.down.sampled <- train(is_exciting ~ ., data = training.down.sampled[, -a],
               method = "rf",
               trControl = ctrl,
               # ntree = 1500,
               # tuneLength = 5,
               metric = "ROC")

# Stochastic Gradient Boosting
set.seed(1410)
btFit <- train(is_exciting ~., data=training[, -a], method="gbm", verbose=F) 

set.seed(1410)
btFit.smoted <- train(is_exciting ~., data=training.smoted[, -a], method="gbm", verbose=F) 

set.seed(1410)
btFit.down.sampled <- train(is_exciting ~., data=training.down.sampled[, -a], method="gbm", verbose=F) 

### Boosted Classification Trees
set.seed(1410)
ada.down.sampled <- train(is_exciting ~ .,
               data = training.down.sampled[, -a],
               method = "ada",
               trControl = ctrl,
               metric = "ROC")

### SVM Radial
set.seed(1410)
svmRadial.down.sampled <- train(is_exciting ~ .,
               data = training.down.sampled[, -a],
               method = "svmRadial",
               trControl = ctrl,
               metric = "ROC")

### Bagged CART
set.seed(1410)
treebag.down.sampled <- train(is_exciting ~ .,
               data = training.down.sampled[, -a],
               method = "treebag",
               trControl = ctrl,
               metric = "ROC")

### Logistic Regression
set.seed(1410)
# ROC of 0.587
lrFit <- train(is_exciting ~ .,
               data = training[, -a],
               method = "glm",
               trControl = ctrl,
               metric = "ROC")

set.seed(1410)
lrFit.smoted <- train(is_exciting ~ .,
               data = training.smoted[, -a],
               method = "glm",
               trControl = ctrl,
               metric = "ROC")

# AUC 0.592
set.seed(1410)
lrFit.down.sampled <- train(is_exciting ~ .,
               data = training.down.sampled[, -a],
               method = "glm",
               trControl = ctrl,
               metric = "ROC")

# now want to estimate AUC using evaluation set
library(pROC)
fit.list <- list(btFit, btFit.down.sampled, btFit.smoted, lrFit, lrFit.smoted,
                 lrFit.down.sampled, rfFit.down.sampled, svmRadial.down.sampled,
                 treebag.down.sampled)
find.auc <- function(x) {
  r <- predict(x, newdata=evaluation[,-37], type="prob")[,2]
  a <- roc(evaluation$is_exciting, r,
                          levels=rev(levels(evaluation$is_exciting)))
  a$auc[1]
}
results <- lapply(fit.list, find.auc)
rm(fit.list)

r <- predict(btFit.down.sampled, newdata=scoring, type="prob")[,2]
df <- data.frame(projectid=scoring$projectid, is_exciting=r)
write.csv(df, file="~/temp/donorschoose_sub.csv")
