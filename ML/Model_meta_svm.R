####################################################################
#####################################################################
# Stacking of several models, using SVM as meta model
#####################################################################
#####################################################################

    rm ( list = ls ( ))
    
    Sys.setlocale("LC_TIME", "English")
    
    library(data.table)
    library(plyr)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library ( xgboost )
    library(Matrix)
    library(caret)
    library(snow)
    library(rlecuyer)
    library(randomForest)
    library(e1071)
    
    # meta data loading
    metadata <- readRDS(file='data/metadata.rds')
    metadata_train <-  xgb.DMatrix( 'data/xgb.DMatrix.metadata.data')
    
    
    
    # aggregation with SVM
    # error function is the inverse of AUC
    
    
    # AUC of original models
    library(ROCR)
    inverse.auc <- function(actual,pred) { auc.tmp <- performance( prediction(predictions= pred , 
                                                                        labels= actual , 
                                                                        label.ordering = NULL),"auc" )
    
                                    return( as.numeric(1/auc.tmp@y.values) )
                                    }
    
    
    
    
    # Training SVM Models with Caret
    library(kernlab)       # support vector machine 
    library(pROC)	       # plot the ROC curves
    
    
    # First pass
    set.seed(1492)
    
    # First time we explore only C
   # at equal length intervals
     
    
    # Setup for cross validation
    ctrl <- trainControl(method="cv",   # 10fold cross validation
                         number=5,		    # do 5 repititions of cv
                         summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                         classProbs=TRUE,
                         allowParallel = TRUE)
    
    library(snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    #Train and Tune the SVM
    svm.tune <- train(QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                      data = metadata[tune_idx,],
                      method = "svmRadial",   # Radial kernel
                      tuneLength = 9,					# 9 values of the cost function
                      preProc = c("center","scale"),  # Center and scale data
                      metric= "ROC",
                      trControl= ctrl)
    
    stopCluster(cl)
    
    
    
    svm.tune
    # C= 0.25 ROC 0.9203741 sigma=166
    
    
    # Use the expand.grid to specify the search space	
    # First time we explore only C
    grid <- expand.grid(sigma = 166,
                        C = c(0.05,0.15,0.25)
    )
    
    
    
    cl <- makeCluster(n_clust , type= "SOCK")
    
    #Train and Tune the SVM
    svm.tune <- train(QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                      data = metadata[tune_idx,],
                      method = "svmRadial",   # Radial kernel
                      preProc = c("center","scale"),  # Center and scale data
                      metric= "ROC",
                      trControl= ctrl,
                      tuneGrid= grid)
    
    stopCluster(cl)
    #C=0.05, sigma=166, ROC= 0.9332612
    
    
    # Use the expand.grid to specify the search space	
    # First time we explore only C
    grid <- expand.grid(sigma = 166,
                        C = c(0.005,0.01,0.025)
    )

    cl <- makeCluster(n_clust , type= "SOCK")
    
    #Train and Tune the SVM
    svm.tune <- train(QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                      data = metadata[tune_idx,],
                      method = "svmRadial",   # Radial kernel
                      preProc = c("center","scale"),  # Center and scale data
                      metric= "ROC",
                      trControl= ctrl,
                      tuneGrid= grid)
    
    stopCluster(cl)
    #C=0.005, sigma=166, ROC= 0.9474257
    
    
    
    grid <- expand.grid(sigma = 166,
                        C = c(0.0005 , 0.001 , 0.0025 )
    )
    
    cl <- makeCluster(n_clust , type= "SOCK")
    #Train and Tune the SVM
    svm.tune <- train(QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                      data = metadata[tune_idx,],
                      method = "svmRadial",   # Radial kernel
                      preProc = c("center","scale"),  # Center and scale data
                      metric= "ROC",
                      trControl= ctrl,
                      tuneGrid= grid)
    
    stopCluster(cl)
    #C=0.0025, sigma=166, ROC= 0.9464027
    
    
    
    grid <- expand.grid(sigma = c(120,166,200),
                        C = c(0.005  )
    )
    
    
    
    cl <- makeCluster(n_clust , type= "SOCK")
    
    #Train and Tune the SVM
    svm.tune <- train(QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                      data = metadata[tune_idx,],
                      method = "svmRadial",   # Radial kernel
                      preProc = c("center","scale"),  # Center and scale data
                      metric= "ROC",
                      trControl= ctrl,
                      tuneGrid= grid)
    
    stopCluster(cl)
    #C=0.005 , sigma=166, ROC= 0.9487825
    
    
    
    grid <- expand.grid(sigma = c(150,166,180),
                        C = c(0.005  )
    )
    
    cl <- makeCluster(n_clust , type= "SOCK")
    
    #Train and Tune the SVM
    svm.tune <- train(QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                      data = metadata[tune_idx,],
                      method = "svmRadial",   # Radial kernel
                      preProc = c("center","scale"),  # Center and scale data
                      metric= "ROC",
                      trControl= ctrl,
                      tuneGrid= grid)
    
    stopCluster(cl)
    #C=0.005 , sigma=180, ROC= 0.9488573
    
    
    grid <- expand.grid(sigma = c(140,145,185,190),
                        C = c(0.005  )
    )
    
    cl <- makeCluster(n_clust , type= "SOCK")
    
    #Train and Tune the SVM
    svm.tune <- train(QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                      data = metadata[tune_idx,],
                      method = "svmRadial",   # Radial kernel
                      preProc = c("center","scale"),  # Center and scale data
                      metric= "ROC",
                      trControl= ctrl,
                      tuneGrid= grid)
    
    stopCluster(cl)
    
    # selected SVm
    # C= 0.005 , sigma= 180
    
    
    # train selected model
    # perform a grid search
    meta.svm <- ksvm( QuoteConversion_Flag ~ rf_fit+xgb_1_fit+xgb_2_fit ,  
                       data = metadata ,
                       scale = TRUE, 
                       type = "C-svc" , 
                       kernel = "rbfdot",
                      kpar=list(sigma= 180),
                      C= 0.005,
                      fit=FALSE,      #whether the fitted values should be computed and included in the model
                      prob.model=TRUE #builds a model for calculating class probabilities 
                      )

        
    # load data for submission
    rf_submission <- read.table(file="submissions/rf2.csv",sep=",",header=TRUE)
    xgb1_submission <- read.table(file="submissions/xgb1.csv",sep=",",header=TRUE)
    xgb2_submission <- read.table(file="submissions/xgb2.csv",sep=",",header=TRUE)
    metadata_test <- data.frame( rf_fit= rf_submission[,2] , 
                                 xgb1_fit= xgb1_submission[,2] ,  
                                 xgb2_fit= xgb2_submission[,2] )

    colnames(metadata_test) <- colnames(metadata)[-1]  # for compatibility
    meta_submission_svm_1 <- predict ( meta.svm , newdata= metadata_test , type= "probabilities" )[,2]
    meta_submission_svm_1 <- data.frame( QuoteNumber= rf_submission[,1] ,  
                              QuoteConversion_Flag= meta_submission_svm_1  )
    write.table(x= meta_submission_svm_1 , file="submissions/meta_svm1.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
  # AUC 0.96011  very close to expected value
    
    
    
    library(corrplot)
    corrplot::corrplot(cor(metadata_test))
#####################################################################
#####################################################################