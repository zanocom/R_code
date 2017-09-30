####################################################################
#####################################################################
# Codice per produrre previsione con metodo Xtreme Gradient Boosting

# Il lavoro viene effettuato in parallelo
# Il codice carica il dataset pronto per l'analisi

#####################################################################
#####################################################################
    
    rm ( list = ls ( ))
    
    Sys.setlocale("LC_TIME", "English")
    
    library(data.table)
    library(plyr)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library (xgboost )
    library(Matrix)
    library(caret)

    
    train <- readRDS(file ="data/train_step_3.rds")
    test <- readRDS(file= "data/test_step_3.rds")
    
    train <- readRDS(file ="data/train_step_3_simple.rds")
    test <- readRDS(file= "data/test_step_3_simple.rds")
    
    dtrain <- xgb.DMatrix('data/xgb.DMatrix.train.simple.data')
    dtest <- xgb.DMatrix('data/xgb.DMatrix.test.simple.data')
    
    train <- tbl_df(train)
    test <- tbl_df(test)
    train$QuoteConversion_Flag <- factor(train$QuoteConversion_Flag)

    
#######################################
# Output should be in probability score
    str(train )    
    summary ( train )

# Per ogni fold identifico train e test dataset di cross validation
# Per ogni parametro della griglia lancio 4 random forest in parallelo
    
    set.seed(823)
    idx_quotenumber <- which(colnames(train)=="QuoteNumber")
    train_subset <- train [ sample(nrow(train) ,size=25e3 ) ,  ]
    
    
#  Convert data in Dense Matrix    
# In order to do so convert factor and characheters into numerical
    which( unlist(lapply(train_subset , is.character)))
    train_subset$Field10 <- as.numeric(train_subset$Field10)
    
    # create dummy variable for factors
    # but only for those with charachter levels
    factor_var_index <-  which( unlist(lapply(train_subset , is.factor)))
    factor_df <- train_subset[,factor_var_index]
    
    factor_var_numeric_index <-  which(lapply(factor_df , function(x) sum( is.na(as.numeric(levels(x))) ) )==0 )
    factor_df_numeric <- data.frame( lapply(factor_df[,factor_var_numeric_index] , function(x) as.numeric(levels(x))[as.numeric(x)] ))
    factor_df <- factor_df[,-factor_var_numeric_index]

    dmy <- dummyVars(" ~ .", data=factor_df)
    factor_df_dummy <- predict( dmy, newdata=factor_df )
    
    
    # identify factor variable with numeric levels
    train_subset <- train_subset[,-factor_var_index]
    train_subset <- cbind(train_subset,factor_df_numeric,factor_df_dummy)
    train_subset$Original_Quote_Date <- as.numeric(train_subset$Original_Quote_Date)
    
    
    rm(factor_df,factor_df_numeric,factor_df_dummy)
    gc()
    
    
    dtrain <- xgb.DMatrix( data= as.matrix(train_subset[,-which(colnames(train_subset)=="QuoteConversion_Flag")]) ,
                           label=as.numeric(train_subset$QuoteConversion_Flag))
    
    xgb.DMatrix.save(dtrain, 'data/xgb.DMatrix.data')
    dtrain <- xgb.DMatrix('data/xgb.DMatrix.data')
    
    xgb.DMatrix.save(dtrain, 'data/xgb.DMatrix.simple_subset.data')
    
    
  
  #######################  
  # Grid Search
    
    # set up the cross-validated hyper-parameter search
    xgb_grid_1 = expand.grid(
      nrounds = 1000,
      eta = c(1e-4,1e-3,1e-2,1e-1 ),
      max_depth = c(2, 4, 6, 8, 10 , 12 ),
      gamma=0, 
      colsample_bytree=1, 
      min_child_weight=1
    )
    
    # pack the training control parameters
    xgb_trcontrol_1 = trainControl(
      method = "cv",
      number = 5,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all", # save losses across all models
      classProbs = TRUE, # set to TRUE for AUC to be computed
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
    )
    
    library (snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    
    # train the model for each parameter combination in the grid, 
    #   using CV to evaluate
    xgb_train_1 = train(
      x = as.matrix( train_subset[,-which(colnames(train_subset)=="QuoteConversion_Flag")] ),
      y = factor( train_subset$QuoteConversion_Flag , labels = c("N","Y") ),
      trControl = xgb_trcontrol_1,
      tuneGrid = xgb_grid_1,
      method = "xgbTree"
    )
    
    stopCluster(cl) 
    
    
    # scatter plot of the AUC against max_depth and eta
    ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
      scale_size_area(max_size = 10)+
      geom_point() + 
      theme_bw() + 
      scale_size_continuous(guide = "none")  
    
    
    ggplot(xgb_train_1$results, aes(x= 1:nrow(xgb_train_1$results), y = ROC-ROCSD)) + 
      geom_bar(stat = "identity") + 
      theme_bw() + 
      coord_cartesian(ylim=c(0.75, 1))+
      scale_y_continuous(breaks=seq(0.75, 1, 0.05))
    
# model selection
    # max AUC
    xgb_train_1$results[which.max(xgb_train_1$results$ROC ),]
#     eta max_depth nrounds       ROC      
#     0.01         6    1000 0.9550091

        
    ########
    # given best parameters
    # IDentify optimal number of trees 
    
    
    xgb_params_1 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.01,                                                                  # learning rate
      max.depth = 6,                                                               # max tree depth
      eval_metric = "auc"                                                          # evaluation/loss metric
    )
    
    xgb_cv_1 = xgb.cv(params = xgb_params_1,
                      data = dtrain,
                      #label = df_train$SeriousDlqin2yrs,
                      nrounds = 3000, 
                      nfold = 5,            # number of folds in K-fold
                      prediction = TRUE,    # return the prediction using the final model 
                      showsd = TRUE,       # standard deviation of loss across folds
                      stratified = TRUE,   # sample is unbalanced; use stratified sampling
                      verbose = TRUE,
                      print.every.n = 1, 
                      early.stop.round = 10
    )
    
    # plot the AUC for the training and testing samples
    xgb_cv_1$dt %>%
      select(-contains("std")) %>%
      mutate(IterationNum = 1:n()) %>%
      gather(TestOrTrain, AUC, -IterationNum) %>%
      ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
      geom_line() + 
      theme_bw()
    #     eta max_depth nrounds             
    #     0.01         6    1000 
    # 900 rounds selected 
    
    
  
      
    # Extending grid search given previous results
    ###
    # set up the cross-validated hyper-parameter search
    xgb_grid_2 = expand.grid(
      nrounds = 1000,
      eta = c( 0.005,0.01, 0.015 , 0.02),
      max_depth = c(4,6,8),
      gamma = 1, 
      colsample_bytree=1, 
      min_child_weight=1
    )
    
    
    
    library (snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    
    # train the model for each parameter combination in the grid, 
    #   using CV to evaluate
    xgb_train_2 = train(
      x = as.matrix( train_subset[,-which(colnames(train_subset)=="QuoteConversion_Flag")] ),
      y = factor( train_subset$QuoteConversion_Flag , labels = c("N","Y") ),
      trControl = xgb_trcontrol_1,
      tuneGrid = xgb_grid_2,
      method = "xgbTree"
    )
    
    stopCluster(cl) 
    
    # scatter plot of the AUC against max_depth and eta
    ggplot(xgb_train_2$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
      scale_size_area(max_size = 10)+
      geom_point() + 
      theme_bw() + 
      scale_size_continuous(guide = "none")  
    
    
    ggplot(xgb_train_2$results, aes(x= 1:nrow(xgb_train_2$results), y = ROC-ROCSD)) + 
      geom_bar(stat = "identity") + 
      theme_bw() + 
      coord_cartesian(ylim=c(0.75, 1))+
      scale_y_continuous(breaks=seq(0.75, 1, 0.05))
    
    # model selection
    # max AUC
    xgb_train_2$results[which.max(xgb_train_2$results$ROC ),]
    #   eta max_depth gamma colsample_bytree min_child_weight nrounds       ROC      Sens      Spec       ROCSD      SensSD
    # 0.01        6     1                1                1    1000 0.955403
    # 
    
    
    ########
    # given best parameters
    # IDentify optimal number of trees 
    # for each optimal combination
    
    xgb_params_2 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.01,                                                                  # learning rate
      max.depth = 6,                                                               # max tree depth
      eval_metric = "auc" ,
      nthread = 4 # evaluation/loss metric
    )
    
    xgb_cv_2 = xgb.cv(params = xgb_params_2,
                      data = dtrain,
                      #label = df_train$SeriousDlqin2yrs,
                      nrounds = 3000, 
                      nfold = 5,            # number of folds in K-fold
                      prediction = TRUE,    # return the prediction using the final model 
                      showsd = TRUE,       # standard deviation of loss across folds
                      stratified = TRUE,   # sample is unbalanced; use stratified sampling
                      verbose = TRUE,
                      print.every.n = 1, 
                      early.stop.round = 10
    )
    
    # plot the AUC for the training and testing samples
    xgb_cv_2$dt %>%
      select(-contains("std")) %>%
      mutate(IterationNum = 1:n()) %>%
      gather(TestOrTrain, AUC, -IterationNum) %>%
      ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
      geom_line() + 
      theme_bw()
    #   eta max_depth gamma colsample_bytree min_child_weight 
    # 0.01        6     1                1                1    
   # 956 rounds selected 
    #test-auc:0.9548
    
   
    
    
    # Extending grid search given previous results
    ###
    # set up the cross-validated hyper-parameter search
    xgb_grid_3 = expand.grid(
      nrounds = 1000,
      eta = c(  0.015 , 0.02 ),
      max_depth = c(4,6,8,10),
      gamma = 1, 
      colsample_bytree=1, 
      min_child_weight=1
    )
    
    # pack the training control parameters
    xgb_trcontrol_1 = trainControl(
      method = "cv",
      number = 5,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all", # save losses across all models
      classProbs = TRUE, # set to TRUE for AUC to be computed
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
    )
    
    
    library (snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    # train the model for each parameter combination in the grid, 
    #   using CV to evaluate
    xgb_train_3 = train(
      x = as.matrix(train_subset[,-115]),
      y = factor(train_subset$QuoteConversion_Flag , labels = c("N","Y")),
      trControl = xgb_trcontrol_1,
      tuneGrid = xgb_grid_3,
      method = "xgbTree"
    )
    
    stopCluster(cl) 
    
    # scatter plot of the AUC against max_depth and eta
    ggplot(xgb_train_3$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
      scale_size_area(max_size = 10)+
      geom_point() + 
      theme_bw() + 
      scale_size_continuous(guide = "none")  
    
    
    ggplot(xgb_train_3$results, aes(x= 1:nrow(xgb_train_3$results), y = ROC-ROCSD)) + 
      geom_bar(stat = "identity") + 
      theme_bw() + 
      coord_cartesian(ylim=c(0.9, 1))+
      scale_y_continuous(breaks=seq(0.9, 1, 0.025))
    
    # model selection
    # max AUC
    xgb_train_3$results[which.max(xgb_train_3$results$ROC ),]
    #   eta max_depth gamma colsample_bytree min_child_weight nrounds       ROC      Sens      Spec       ROCSD      SensSD
    # 0.015         4     1                1                1    1000 0.9526307
    # 
    
    
    xgb_params_3 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.015,                                                                  # learning rate
      max.depth = 6,                                                               # max tree depth
      eval_metric = "auc" ,
      nthread = 4 # evaluation/loss metric
    )
    
    xgb_cv_3 = xgb.cv(params = xgb_params_3,
                      data = dtrain,
                      #label = df_train$SeriousDlqin2yrs,
                      nrounds = 5000, 
                      nfold = 5,            # number of folds in K-fold
                      prediction = FALSE,    # return the prediction using the final model 
                      showsd = TRUE,       # standard deviation of loss across folds
                      stratified = TRUE,   # sample is unbalanced; use stratified sampling
                      verbose = TRUE,
                      print.every.n = 1, 
                      early.stop.round = 10
    )
    
    # plot the AUC for the training and testing samples
    xgb_cv_3 %>%
      select(-contains("std")) %>%
      mutate(IterationNum = 1:n()) %>%
      gather(TestOrTrain, AUC, -IterationNum) %>%
      ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
      geom_line() + 
      theme_bw()
    #   eta max_depth gamma colsample_bytree min_child_weight 
    # 0.015         4     1                1                1
    # 1014 rounds selected 
    
  
      
    
    
    # Extending grid search given previous results
    ###
    # set up the cross-validated hyper-parameter search
    xgb_grid_1 = expand.grid(
      nrounds = 1000,
      eta = c( 0.0125, 0.015 , 0.0175),
      max_depth = c(5,6,7),
      gamma = 1, 
      colsample_bytree=1, 
      min_child_weight=1
    )
    
    # pack the training control parameters
    xgb_trcontrol_1 = trainControl(
      method = "cv",
      number = 5,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all", # save losses across all models
      classProbs = TRUE, # set to TRUE for AUC to be computed
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
    )
    
    library (snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    
    # train the model for each parameter combination in the grid, 
    #   using CV to evaluate
    xgb_train_2 = train(
      x = as.matrix(train_subset[,-115]),
      y = factor(train_subset$QuoteConversion_Flag , labels = c("N","Y")),
      trControl = xgb_trcontrol_1,
      tuneGrid = xgb_grid_1,
      method = "xgbTree"
    )
    
    stopCluster(cl) 
    
    
    
    ggplot(xgb_train_2$results, aes(x= 1:nrow(xgb_train_2$results), y = ROC-ROCSD)) + 
      geom_bar(stat = "identity") + 
      theme_bw() + 
      coord_cartesian(ylim=c(0.75, 1))+
      scale_y_continuous(breaks=seq(0.75, 1, 0.05))
    
    # model selection
    # max AUC
    xgb_train_2$results[which.max(xgb_train_2$results$ROC ),]
    #   eta max_depth gamma colsample_bytree min_child_weight nrounds       ROC      Sens      Spec       ROCSD      SensSD
    # 0.015         5     1                1                1    1000 0.9530
    # 
    
    
    
    
    
    
    
    
    
    ########
    # given best parameters
    # IDentify optimal number of trees 
    # for each optimal combination
    
    xgb_params_2 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.015,                                                                  # learning rate
      max.depth = 5,                                                               # max tree depth
      eval_metric = "auc" ,
      nthread = 4 # evaluation/loss metric
    )
    
    xgb_cv_2 = xgb.cv(params = xgb_params_2,
                      data = dtrain,
                      #label = df_train$SeriousDlqin2yrs,
                      nrounds = 3000, 
                      nfold = 5,            # number of folds in K-fold
                      prediction = TRUE,    # return the prediction using the final model 
                      showsd = TRUE,       # standard deviation of loss across folds
                      stratified = TRUE,   # sample is unbalanced; use stratified sampling
                      verbose = TRUE,
                      print.every.n = 1, 
                      early.stop.round = 10
    )
    
    
    #   eta max_depth gamma colsample_bytree min_child_weight 
    # 0.015         5     1                1                1    
    # 2550 rounds selected 
    
    
    
    # Tuning Method suggested by Owen Zhang
    # http://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1
    ###
    # set up the cross-validated hyper-parameter search
    xgb_grid_5 = expand.grid(
      nrounds = 100,
      eta = c( 0.01, 0.02),
      max_depth = c(4,6,8,10,12),
      gamma = c(0.7,1,1.3) , 
      colsample_bytree=c(0.3,0.4,0.5,0.6) , 
      #min_child_weight=1/sqrt(mean(c(0,1)[as.numeric(train$QuoteConversion_Flag)])) 
      min_child_weight= 1 
    )
    
    # pack the training control parameters
    xgb_trcontrol_5 = trainControl(
      method = "cv",
      number = 5,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all", # save losses across all models
      classProbs = TRUE, # set to TRUE for AUC to be computed
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
    )
    
    
      
      library (snow)
      n_clust <- 4
      cl <- makeCluster(n_clust , type= "SOCK")
      
      
      # train the model for each parameter combination in the grid, 
      #   using CV to evaluate
      xgb_train_5 = train(
        x = as.matrix(train_subset[,-which(colnames(train_subset)=="QuoteConversion_Flag")]),
        y = factor(train_subset$QuoteConversion_Flag , labels = c("N","Y")),
        trControl = xgb_trcontrol_5,
        tuneGrid = xgb_grid_5,
        method = "xgbTree"
      )
      
      stopCluster(cl) 
      
      
      ggplot(xgb_train_5$results, aes(x= 1:nrow(xgb_train_5$results), y = ROC)) + 
        geom_bar(stat = "identity") + 
        geom_bar(data= xgb_train_5$results, aes(x= 1:nrow(xgb_train_5$results), y = ROC-ROCSD) , fill=NA , colour="white" , stat = "identity") + 
        theme_bw() + 
        coord_cartesian(ylim=c(0.75, 1))+
        scale_y_continuous(breaks=seq(0.75, 1, 0.05))
      
      
      xgb_train_5$results[which.max(xgb_train_5$results$ROC ),]
      xgb_train_5$results[order(xgb_train_5$results$ROC , decreasing = TRUE),][1:10,]
      xgb_train_5$results[order(xgb_train_5$results$ROC-xgb_train_5$results$ROCSD , decreasing = TRUE),][1:10,]
      
      
      ###
    
        xgb_grid_5_bis = expand.grid(
          nrounds = 100,
          eta = c(  0.02 , 0.03 , 0.04 ),
          max_depth = c(10,12,14),
          gamma = 1, 
          colsample_bytree=c( 0.6 , 0.7 , 0.8 ) , 
          min_child_weight=1/sqrt(mean(c(0,1)[as.numeric(train$QuoteConversion_Flag)]))
        )
        
        xgb_grid_5_bis = expand.grid(
          nrounds = 100,
          eta = c(  0.02 , 0.03  ),
          max_depth = 10 ,
          gamma = 1, 
          colsample_bytree=c( 0.6  , 0.8 , 1) , 
          min_child_weight=1
        )
        
    library (snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    
    # train the model for each parameter combination in the grid, 
    #   using CV to evaluate
    xgb_train_5_bis = train(
      x = as.matrix(train_subset[,-which(colnames(train_subset)=="QuoteConversion_Flag")]),
      y = factor(train_subset$QuoteConversion_Flag , labels = c("N","Y")),
      trControl = xgb_trcontrol_5,
      tuneGrid = xgb_grid_5_bis,
      method = "xgbTree"
    )
    
    stopCluster(cl) 
    
    
    
    ggplot(xgb_train_5_bis$results, aes(x= 1:nrow(xgb_train_5_bis$results), y = ROC)) + 
      geom_bar(stat = "identity") + 
      geom_bar(data= xgb_train_5_bis$results, aes(x= 1:nrow(xgb_train_5_bis$results), y = ROC-ROCSD) , fill=NA , colour="white" , stat = "identity") + 
      theme_bw() + 
      coord_cartesian(ylim=c(0.75, 1))+
      scale_y_continuous(breaks=seq(0.75, 1, 0.05))
    
    
    xgb_train_5_bis$results[which.max(xgb_train_5_bis$results$ROC ),]
    xgb_train_5_bis$results[order(xgb_train_5_bis$results$ROC , decreasing = TRUE),][1:10,]
    
    ####
    
    xgb_grid_5_tris = expand.grid(
      nrounds = 100,
      eta = c(  0.05 , 0.06 , 0.07 , 0.08 ),
      max_depth = c(10),
      gamma = 1, 
      colsample_bytree=0.6, 
      min_child_weight=1/sqrt(mean(c(0,1)[as.numeric(train$QuoteConversion_Flag)]))
    )
    
    
    xgb_grid_5_tris = expand.grid(
      nrounds = 100,
      eta = c(  0.04 , 0.06  , 0.08 ),
      max_depth = 10 ,
      gamma = 1, 
      colsample_bytree=c( 0.6   ) , 
      min_child_weight=1
    )
    
    
    
    library (snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    
    # train the model for each parameter combination in the grid, 
    #   using CV to evaluate
    xgb_train_5_tris = train(
      x = as.matrix(train_subset[,-which(colnames(train_subset)=="QuoteConversion_Flag")]),
      y = factor(train_subset$QuoteConversion_Flag , labels = c("N","Y")),
      trControl = xgb_trcontrol_5,
      tuneGrid = xgb_grid_5_tris ,
      method = "xgbTree"
    )
    
    stopCluster(cl) 
    
    
    
    ggplot(xgb_train_5_tris$results, aes(x= 1:nrow(xgb_train_5_tris$results), y = ROC)) + 
      geom_bar(stat = "identity") + 
      geom_bar(data= xgb_train_5_tris$results, aes(x= 1:nrow(xgb_train_5_tris$results), y = ROC-ROCSD) , fill=NA , colour="white" , stat = "identity") + 
      theme_bw() + 
      coord_cartesian(ylim=c(0.75, 1))+
      scale_y_continuous(breaks=seq(0.75, 1, 0.05))
    
    
    xgb_train_5_tris$results[which.max(xgb_train_5_tris$results$ROC ),]
    xgb_train_5_tris$results[order(xgb_train_5_tris$results$ROC , decreasing = TRUE),]
    
    
    ###
    
    
    
    xgb_grid_5_quater = expand.grid(
      nrounds = 100,
      eta = c(  0.07  , 0.09 ),
      max_depth = 10 ,
      gamma = 1, 
      colsample_bytree=c( 0.6   ) , 
      min_child_weight=1
    )
    
    
    
    library (snow)
    n_clust <- 4
    cl <- makeCluster(n_clust , type= "SOCK")
    
    
    # train the model for each parameter combination in the grid, 
    #   using CV to evaluate
    xgb_train_5_quater = train(
      x = as.matrix(train_subset[,-which(colnames(train_subset)=="QuoteConversion_Flag")]),
      y = factor(train_subset$QuoteConversion_Flag , labels = c("N","Y")),
      trControl = xgb_trcontrol_5,
      tuneGrid = xgb_grid_5_quater ,
      method = "xgbTree"
    )
    
    stopCluster(cl) 
    
    str(xgb_train_5_quater)
    xgb_train_5_quater$results[order(xgb_train_5_quater$results$ROC , decreasing = TRUE),]
    
    
    ###
    
    
    # model selection
    # max AUC
    saveRDS(
    rbind(
    xgb_train_5$results[which.max(xgb_train_5$results$ROC ),] ,           # 0.956044
    xgb_train_5_bis$results[which.max(xgb_train_5_bis$results$ROC ),] ,   # 0.9576591
    xgb_train_5_tris$results[which.max(xgb_train_5_tris$results$ROC ),] , # 0.9579737
    xgb_train_5_quater$results[which.max(xgb_train_5_quater$results$ROC ),]  # 0.9579737
    ) , file="data/params_archive.rds" ) 
    
    
    
    # define optimal number of trees for selected model
    xgb_params_5 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.08,                                                                  # learning rate
      max.depth = 10,                                                               # max tree depth
      eval_metric = "auc" ,
      gamma = 1, 
      colsample_bytree=0.6, 
  #    min_child_weight=1/sqrt(mean(c(0,1)[as.numeric(train$QuoteConversion_Flag)])) ,
      min_child_weight=1,
      
      nthread = 4 # evaluation/loss metric
    )
    
    xgb_cv_5 = xgb.cv(params = xgb_params_5,
                      data = dtrain,
                      #label = df_train$SeriousDlqin2yrs,
                      nrounds = 3000, 
                      nfold = 5,            # number of folds in K-fold
                      prediction = TRUE,    # return the prediction using the final model 
                      showsd = TRUE,       # standard deviation of loss across folds
                      stratified = TRUE,   # sample is unbalanced; use stratified sampling
                      verbose = TRUE,
                      print.every.n = 1, 
                      early.stop.round = 10
    )
    
    
    
  #######################  
# Submission     

        
    xgb_params_1 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.08,                                                                  # learning rate
      max.depth = 10,                                                               # max tree depth
      eval_metric = "auc"   , # evaluation/loss metric
      gamma = 1, 
      colsample_bytree=0.6, 
      min_child_weight=1 
    )
    
    set.seed(73)
    
    xgb_fit = xgboost(data = dtrain,
                    params = xgb_params_1,
                    nrounds = 100,                                                 # max number of trees to build
                    verbose = TRUE,                                         
                    print.every.n = 1,
                    early.stop.round = 10                                          # stop if no improvement within 10 trees
    )
    # train-auc:0.998438
    # AUC 0.96091
    
    
    test_prediction <- predict ( xgb_fit , newdata= dtest )
    submission <- data.frame( QuoteNumber= testQuoteNumber ,  
                              QuoteConversion_Flag= test_prediction  )
    str(submission)
    
    write.table(x= submission , file="submissions/xgb1.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    write.table(x= submission , file="submissions/xgb8.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    write.table(x= submission , file="submissions/xgb9.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    
    # sample test AUC 0.957 ranking 567
    # xgb9 = 0.96057

        
    
    xgb_params_2 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.015,                                                                  # learning rate
      max.depth = 6,                                                               # max tree depth
      eval_metric = "auc" ,  # evaluation/loss metric
      nthread=4 
    )
    
    
    xgb_fit_2 = xgboost(data = dtrain,
                      #label = df_train$SeriousDlqin2yrs,
                      params = xgb_params_2,
                      nrounds = 2550,                                                 # max number of trees to build
                      verbose = TRUE,                                         
                      print.every.n = 1,
                      early.stop.round = 10                                          # stop if no improvement within 10 trees
    )
    
    
    test_prediction <- predict ( xgb_fit_2 , newdata= dtest )
    submission_xgb2 <- data.frame( QuoteNumber= testQuoteNumber ,  
                              QuoteConversion_Flag= test_prediction  )
    str(submission_xgb2)
    
    write.table(x= submission_xgb2 , file="submissions/xgb2.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    write.table(x= submission_xgb2 , file="submissions/xgb10.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    
    # sample test AUC 0.96306
    # xgb10 = 0.96042
    
    
    
    xgb_params_3 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.015,                                                                  # learning rate
      max.depth = 4,                                                               # max tree depth
      eval_metric = "auc" ,  # evaluation/loss metric
      nthread=4 
    )
    
    
    xgb_fit_3 = xgboost(data = dtrain,
                        #label = df_train$SeriousDlqin2yrs,
                        params = xgb_params_3,
                        nrounds = 1000,                                                 # max number of trees to build
                        verbose = TRUE,                                         
                        print.every.n = 1,
                        early.stop.round = 10                                          # stop if no improvement within 10 trees
    )
    
    
    test_prediction <- predict ( xgb_fit_3 , newdata= dtest )
    submission_xgb3 <- data.frame( QuoteNumber= test_quote_number ,  
                                   QuoteConversion_Flag= test_prediction  )
    
    write.table(x= submission_xgb3 , file="submissions/xgb3.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    # sample test AUC 0.95660 
    
    
    xgb.save(xgb_fit, 'data/xgb.model')
    xgb.save(xgb_fit_2, 'data/xgb2.model')
    
    
    xgb_params_4 = list(
      objective = "binary:logistic",                                               # binary classification
      eta = 0.015,                                                                  # learning rate
      max.depth = 5,                                                               # max tree depth
      eval_metric = "auc" ,  # evaluation/loss metric
      nthread=4 
    )
    
    
    xgb_fit_4 = xgboost(data = dtrain,
                        #label = df_train$SeriousDlqin2yrs,
                        params = xgb_params_4,
                        nrounds = 910,                                                 # max number of trees to build
                        verbose = TRUE,                                         
                        print.every.n = 1,
                        early.stop.round = 10                                          # stop if no improvement within 10 trees
    )
    
    
    test_prediction <- predict ( xgb_fit_4 , newdata= dtest )
    submission_xgb4 <- data.frame( QuoteNumber= test$QuoteNumber ,  
                                   QuoteConversion_Flag= test_prediction  )
    
    write.table(x= submission_xgb4 , file="submissions/xgb4.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    # sample test AUC 0.95684 
    
    xgb.save(xgb_fit_4, 'data/xgb4.model')
    
    
    ####
    
    
    xgb_fit_5 = xgboost(data = dtrain,
                        #label = df_train$SeriousDlqin2yrs,
                        params = xgb_params_5,
                        nrounds = 125,                                                 # max number of trees to build
                        verbose = TRUE,                                         
                        print.every.n = 1,
                        early.stop.round = 10                                          # stop if no improvement within 10 trees
    )
    
    
    test_prediction <- predict ( xgb_fit_5 , newdata= dtest )
    submission_xgb5 <- data.frame( QuoteNumber= test$QuoteNumber ,  
                                   QuoteConversion_Flag= test_prediction  )
    
    write.table(x= submission_xgb5 , file="submissions/xgb5.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    # sample test AUC 0.95573 
    
    xgb.save(xgb_fit_5, 'data/xgb5.model')
    
    ###
    
    
    xgb_fit_6 = xgboost(data = dtrain,
                        #label = df_train$SeriousDlqin2yrs,
                        params = xgb_params_6,
                        nrounds = 125,                                                 # max number of trees to build
                        verbose = TRUE,                                         
                        print.every.n = 1,
                        early.stop.round = 10                                          # stop if no improvement within 10 trees
    )
    
    
    test_prediction <- predict ( xgb_fit_6 , newdata= dtest )
    submission_xgb5 <- data.frame( QuoteNumber= test$QuoteNumber ,  
                                   QuoteConversion_Flag= test_prediction  )
    
    write.table(x= submission_xgb6 , file="submissions/xgb6.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    # sample test AUC 0.95573 
    
    xgb.save(xgb_fit_5, 'data/xgb6.model')
    
    
    ###
    
    
    
    
    xgb_fit_7 = xgboost(data = dtrain,
                        #label = df_train$SeriousDlqin2yrs,
                        params = xgb_params_7,
                        nrounds =  225 ,                                                 # max number of trees to build
                        verbose = TRUE,                                         
                        print.every.n = 1,
                        early.stop.round = 10                                          # stop if no improvement within 10 trees
    )
    
    
    test_prediction <- predict ( xgb_fit_7 , newdata= dtest )
    submission_xgb7 <- data.frame( QuoteNumber= test$QuoteNumber ,  
                                   QuoteConversion_Flag= test_prediction  )
    
    write.table(x= submission_xgb7 , file="submissions/xgb7.csv" , col.names =TRUE  , row.names = FALSE ,sep="," , dec="." ,quote=FALSE)
    # sample test AUC 0.95629
    
    xgb.save(xgb_fit_7, 'data/xgb7.model')
    
    
    