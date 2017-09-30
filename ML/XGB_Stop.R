# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example
rm(list=ls())
library(readr)
library(xgboost)

#my favorite seed^^
set.seed(1718)

cat("reading the train and test data\n")
train <- readRDS(file ="data/train.rds")
test <- readRDS(file= "data/test.rds")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)


# seperating out the elements of the date column for the train set
train$Original_Quote_Date <- as.Date(train$Original_Quote_Date)
train$week  <- as.integer(format(train$Original_Quote_Date, "%W"))
train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$day <- weekdays(as.Date(train$Original_Quote_Date))

# removing the date column
train <- train[,-c(2)]

# seperating out the elements of the date column for the train set
test$Original_Quote_Date <- as.Date(test$Original_Quote_Date)
test$week  <- as.integer(format(test$Original_Quote_Date, "%W"))
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$day <- weekdays(as.Date(test$Original_Quote_Date))

# removing the date column
test <- test[,-c(2)]


feature.names <- names(train)[c(3:302)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]

nrow(train)
h<-sample(nrow(train),2000)

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
#dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
dtrain<-xgb.DMatrix(data=data.matrix(tra),label=train$QuoteConversion_Flag)

watchlist<-list(val=dval,train=dtrain)

xgb_params_1 <- list(  objective= "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.023, # 0.06, #0.01,
                max_depth           = 6, #changed from default of 8
                subsample           = 0.83, # 0.7
                colsample_bytree    = 0.77 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)


xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = dtrain,
                  #label = df_train$SeriousDlqin2yrs,
                  nrounds = 3000, 
                  nfold = 5,            # number of folds in K-fold
                  prediction = FALSE,    # return the prediction using the final model 
                  showsd = FALSE,       # standard deviation of loss across folds
                  stratified = TRUE,   # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print.every.n = 1, 
                  early.stop.round = 10
                  )



clf <- xgb.train(   params              = xgb_params_1, 
                    data                = dtrain, 
                  #  nrounds             = 20, 
                   nrounds             =  1350, 
                  #  verbose             = 0,  
                    verbose             = 1,
                #  early.stop.round    = 150,
                #  watchlist           = watchlist,
                    maximize            = FALSE
              )

pred1 <- predict(clf, data.matrix(test[,feature.names]))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write.csv(submission,file="submissions/xgb_Shize_stop_3_check.csv",row.names = FALSE)
str(submission)
head(submission)

#test-auc:0.966751