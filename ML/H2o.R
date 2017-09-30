
##################################################
    train <- readRDS(file ="data/train_step_3_bis_mix.rds")
    test <- readRDS(file= "data/test_step_3_bis_mix.rds")
    QuoteConversion_Flag <- readRDS(file="QuoteConversion_Flag.rds")
    QuoteConversion_Flag <- as.factor(QuoteConversion_Flag)
    testQuoteNumber <- readRDS(file="test_quote_number.rds")
    train <- cbind(train , QuoteConversion_Flag)
    
    nfold <- 5
    set.seed(4)
    fold <- sample( rep(1:nfold,length=nrow(train)) )
    
    
    # subset 20 k for model tuning
    subset_size <- 20e3
    set.seed(8)
    idx_subset <- sample( nrow(train) )[1:subset_size]
    training_subset <- train[idx_subset,]
    fold_subset <- fold[idx_subset]
    training_subset$fold <- fold_subset
    
    
    
    colnames(train)
    dim(train)

    library(h2o)
    localH2O = h2o.init(nthreads = -1)

    
    # import dataset in h2o
    train.h2o <- as.h2o( train , destination_frame="train.h2o")
    test.h2o <- as.h2o( test , destination_frame="test.h2o")
    fold.h2o <- as.h2o( fold , destination_frame="fold.h2o")
    train_subset.h2o <- as.h2o( training_subset , destination_frame="train_subset.h2o")
    TQN.h2o <- as.h2o( testQuoteNumber , destination_frame="TQN.h2o")
    
    tuning_params <- expand.grid( max_depth = c(10,20,30),
                                  mtries= c(10,20,30,40) ,
                                  min_rows= c(5,15,25)   )
    
    tuning_results <- numeric(  nrow(tuning_params) )
    
    
        for ( p in 1:nrow(tuning_params) ){
    #    p <- 1
          model <- 
            h2o.randomForest(y = "QuoteConversion_Flag", 
                             x = 1:304, 
                             training_frame = train_subset.h2o ,
                             fold_column = "fold" , 
                             ntrees = 500,
                             max_depth = tuning_params[p,1],
                             mtries= tuning_params[p,2],
                             min_rows = tuning_params[p,3] ,
                             stopping_metric="AUC")
          
          
          tuning_results[p] <- model@model$cross_validation_metrics@metrics$AUC
        }   
    
    
    cbind( tuning_params ,tuning_results) [order(tuning_results,decreasing = TRUE),]
 
 ###      
    
    tuning_params_2 <- expand.grid( max_depth = c(20, 25,30),
                                  mtries= c(50,60) ,
                                  min_rows= c(3,5,7)   )
    
    tuning_results_2 <- numeric(  nrow(tuning_params_2) )
    
    
    for ( p in 1:nrow(tuning_params_2) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "QuoteConversion_Flag", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 500,
                         max_depth = tuning_params_2[p,1],
                         mtries= tuning_params_2[p,2],
                         min_rows = tuning_params_2[p,3] ,
                         stopping_metric="AUC")
      
      
      tuning_results_2[p] <- model@model$cross_validation_metrics@metrics$AUC
    }   
    
    
    
    cbind( tuning_params_2 ,tuning_results_2) [order(tuning_results_2,decreasing = TRUE),]
    
###  
    
    tuning_params_3 <- expand.grid( max_depth = c(30,40),
                                    mtries= c(70,80) ,
                                    min_rows= c(1,3,5)   )
    
    tuning_results_3 <- numeric(  nrow(tuning_params_3) )
    
    
    for ( p in 1:nrow(tuning_params_3) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "QuoteConversion_Flag", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 500,
                         max_depth = tuning_params_3[p,1],
                         mtries=     tuning_params_3[p,2],
                         min_rows =  tuning_params_3[p,3] ,
                         stopping_metric="AUC")
      
      
      tuning_results_3[p] <- model@model$cross_validation_metrics@metrics$AUC
    }   
    
    
    
    cbind( tuning_params_3 ,tuning_results_3) [order(tuning_results_3,decreasing = TRUE),]
    
    ###  
    tuning_params_4 <- expand.grid( max_depth = c(50,60),
                                    mtries= c(70,80) ,
                                    min_rows= c(1,3)   )
    
    tuning_results_4 <- numeric(  nrow(tuning_params_4) )
    
    
    for ( p in 1:nrow(tuning_params_3) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "QuoteConversion_Flag", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 500,
                         max_depth = tuning_params_4[p,1],
                         mtries=     tuning_params_4[p,2],
                         min_rows =  tuning_params_4[p,3] ,
                         stopping_metric="AUC")
      
      
      tuning_results_4[p] <- model@model$cross_validation_metrics@metrics$AUC
    }   
    
    
    
    cbind( tuning_params_4 ,tuning_results_4) [order(tuning_results_4,decreasing = TRUE),]
    
    ###  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    final_model <- 
      h2o.randomForest(y = "QuoteConversion_Flag", 
                       x = 1:304, 
                       training_frame = train.h2o ,
                       ntrees = 500,
                       max_depth = final_params[1],
                       mtries= final_params[2],
                       min_rows = final_params[3] )
    
    
    pred_labels <- h2o.predict(model, test.h2o)
    str(pred_labels)
    
    
    
    
    
    
    
    
    
    
      
      grid <- h2o.grid("randomForest", 
                       y = "QuoteConversion_Flag", 
                       x = 1:304, 
                       training_frame = train_subset.h2o[(fold_subset.h2o!=f)>0] , 
                       validation_frame = train_subset.h2o[fold_subset.h2o==f] , 
                       nfolds= 5 ,
                       hyper_params = list(ntrees = 500,
                                           max_depth = c(10,20,30),
                                           mtries= c(10,20,30) ,
                                           min_rows= c(5,15,25),
                                           stopping_metric="AUC" ) 
      )      
      
    summary(grid)
    
    
    
    
    best_model <- grid@model[[1]]
    best_params <- best_model@model$params
    
    
    
    
    model <- 
    h2o.randomForest(y = "QuoteConversion_Flag", 
                     x = 1:304, 
                     training_frame = train.h2o , 
                     ntrees = c(100,250,500),
                     max_depth = c(10,20,30),
                     mtries= c(10,20,30),
                     nfolds= 5 ,
                     stopping_metric="AUC")
    
    pred_labels <- h2o.predict(model, test.h2o)[,1]
    
    
    #################################
    ## Example of tune grid with for loop
    
    ntree = seq(100,500,100)
    balance_class = c(TRUE,FALSE)
    learn_rate = seq(.05,.4,.05)
    
    parameters = list(ntree = c(), balance_class = c(), learn_rate = c(), r2 = c(), min.r2 = c(), max.r2 = c(), acc = c(), min.acc = c(), max.acc = c(), AUC = c(), min.AUC = c(), max.AUC = c())
    n = 1
    
    mooc.hex = as.h2o(localH2O, mooc[,c("enrollment_id","dropped_out_factor",x.names)])
    for (trees in ntree) {
      for (c in balance_class) {
        for (rate in learn_rate) {
          r2.temp = c(NA,NA,NA)
          acc.temp = c(NA,NA,NA)
          auc.temp = c(NA,NA,NA)
          for (i in 1:3) {
            
            mooc.hex.split = h2o.splitFrame(mooc.hex, ratios=.8)   
            train.gbm = h2o.gbm(x = x.names, y = "dropped_out_factor",  training_frame = mooc.hex.split[[1]],
                                validation_frame = mooc.hex.split[[2]], ntrees = trees, balance_classes = c, learn_rate = rate)
            r2.temp[i] = train.gbm@model$validation_metrics@metrics$r2
            acc.temp[i] = train.gbm@model$validation_metrics@metrics$max_criteria_and_metric_scores[4,3]
            auc.temp[i] = train.gbm@model$validation_metrics@metrics$AUC
          }
          parameters$ntree[n] = trees
          parameters$balance_class[n] = c
          parameters$learn_rate[n] = rate
          parameters$r2[n] = mean(r2.temp)
          parameters$min.r2[n] = min(r2.temp)
          parameters$max.r2[n] = max(r2.temp)
          parameters$acc[n] = mean(acc.temp)
          parameters$min.acc[n] = min(acc.temp)
          parameters$max.acc[n] = max(acc.temp)
          parameters$AUC[n] = mean(auc.temp)
          parameters$min.AUC[n] = min(auc.temp)
          parameters$max.AUC[n] = max(auc.temp)
          n = n+1
        }
      }
    }
    
    
    parameters.df = data.frame(parameters)
    parameters.df[which.max(parameters.df$AUC),]
    
    
    
    
    
    
    
    
    
    
    
    h2o.shutdown(prompt = FALSE)
    
    
    
    
    
    ##################################################
    ##################################################
    
    rm ( list = ls ( ))
    
    Sys.setlocale("LC_TIME", "English")
    
    library(data.table)
    library(plyr)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library (randomForest )
    library(Matrix)
    library(caret)
    
    
    # train <- readRDS( file="data/processed/train_1_adj.rds")
    # test <- readRDS( file="data/processed/test_1_adj.rds")
    train_subset <- readRDS( file="data/processed/train_1_sub_adj.rds")
    # test_ID <- readRDS( file="data/processed/test_ID.rds")
    
    
    # train <- tbl_df(train)
    train_subset <- tbl_df(train_subset)
    # test <- tbl_df(test)
    
    # train$target <- factor(train$target)
    train_subset$target <- factor(train_subset$target)
    
    #######################################
    
    
    nfold <- 5
    set.seed(4)
    fold <- sample( rep(1:nfold,length=nrow(train_subset)) )
    sample_idx <- sample ( nrow(train_subset) )[1:10e3]
    
    # subset 20 k for model tuning
    train_subset$fold <- fold
    
    tuning_params <- expand.grid( max_depth = c(10,20,30,40)    ,
                                  mtries=    seq(5,40,by=5)  ,
                                  min_rows=  seq(5,15,by=5)  )
    
    
    tuning_results <- numeric(  nrow(tuning_params) )
    
    
    library(h2o)
    localH2O = h2o.init(nthreads = 3 , max_mem_size = "5g" )
    
    # Finally, let's run a demo to see H2O at work.
    
    # import dataset in h2o
    train_subset.h2o <- as.h2o( train_subset[sample_idx,] , destination_frame="train_subset.h2o")
    
    
    set.seed(8)
    for ( p in 1:nrow(tuning_params) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "target", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 200,
                         max_depth = tuning_params[p,1],
                         mtries= tuning_params[p,2],
                         min_rows = tuning_params[p,3] ,
                         stopping_metric="logloss")
      
      tuning_results[p] <- model@model$cross_validation_metrics@metrics$logloss
      h2o.rm( "model" )
      #rm( model )
      gc()
    }   
    
    
    cbind( tuning_params ,tuning_results) [order(tuning_results),]
    saveRDS(tuning_results , file="data/rf_tuning_1.rds")
    
    
    
    # best model has depth = 20 , mtries= 24 , min_rows= 5
    
    tuning_params <- expand.grid( max_depth = c(50,60)    ,
                                  mtries=     c(40,50)  ,
                                  min_rows=   c(5,10,15)  )
    
    
    tuning_results <- numeric(  nrow(tuning_params) )
    
    
    set.seed(8)
    for ( p in 1:nrow(tuning_params) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "target", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 200,
                         max_depth = tuning_params[p,1],
                         mtries= tuning_params[p,2],
                         min_rows = tuning_params[p,3] ,
                         stopping_metric="logloss")
      
      tuning_results[p] <- model@model$cross_validation_metrics@metrics$logloss
      h2o.rm( "model" )
      #rm( model )
      gc()
    }   
    
    
    cbind( tuning_params ,tuning_results) [order(tuning_results),]
    
    saveRDS(tuning_results , file="data/rf_tuning_2.rds")
    
    
    ###
    tuning_params <- expand.grid( max_depth = c(50,60)  ,
                                  mtries=     c(55,60,70)  ,
                                  min_rows=   c(10)  )
    
    
    tuning_results <- numeric(  nrow(tuning_params) )
    
    
    set.seed(8)
    for ( p in 1:nrow(tuning_params) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "target", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 200,
                         max_depth = tuning_params[p,1],
                         mtries= tuning_params[p,2],
                         min_rows = tuning_params[p,3] ,
                         stopping_metric="logloss")
      
      tuning_results[p] <- model@model$cross_validation_metrics@metrics$logloss
      h2o.rm( "model" )
      #rm( model )
      gc()
    }   
    
    
    cbind( tuning_params ,tuning_results) [order(tuning_results),]
    saveRDS(tuning_results , file="data/rf_tuning_3.rds")
    
    
    
    
    ###
    tuning_params <- expand.grid( max_depth = c(50)  ,
                                  mtries=     c(80,90)  ,
                                  min_rows=   c(10)  )
    
    
    tuning_results <- numeric(  nrow(tuning_params) )
    
    
    set.seed(8)
    for ( p in 1:nrow(tuning_params) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "target", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 200,
                         max_depth = tuning_params[p,1],
                         mtries= tuning_params[p,2],
                         min_rows = tuning_params[p,3] ,
                         stopping_metric="logloss")
      
      tuning_results[p] <- model@model$cross_validation_metrics@metrics$logloss
      h2o.rm( "model" )
      #rm( model )
      gc()
    }   
    
    
    cbind( tuning_params ,tuning_results) [order(tuning_results),]
    saveRDS(tuning_results , file="data/rf_tuning_4.rds")
    
    
    
    ###
    tuning_params <- expand.grid( max_depth = c(50)  ,
                                  mtries=     c(75,85,100)  ,
                                  min_rows=   c(10)  )
    
    
    tuning_results <- numeric(  nrow(tuning_params) )
    
    
    set.seed(8)
    for ( p in 1:nrow(tuning_params) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "target", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 200,
                         max_depth = tuning_params[p,1],
                         mtries= tuning_params[p,2],
                         min_rows = tuning_params[p,3] ,
                         stopping_metric="logloss")
      
      tuning_results[p] <- model@model$cross_validation_metrics@metrics$logloss
      h2o.rm( "model" )
      #rm( model )
      gc()
    }   
    
    
    cbind( tuning_params ,tuning_results) [order(tuning_results),]
    saveRDS(tuning_results , file="data/rf_tuning_5.rds")
    
    ###
    tuning_params <- expand.grid( max_depth = c(50)  ,
                                  mtries=     c(83,87)  ,
                                  min_rows=   c(10)  )
    
    
    tuning_results <- numeric(  nrow(tuning_params) )
    
    
    set.seed(8)
    for ( p in 1:nrow(tuning_params) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "target", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 200,
                         max_depth = tuning_params[p,1],
                         mtries= tuning_params[p,2],
                         min_rows = tuning_params[p,3] ,
                         stopping_metric="logloss")
      
      tuning_results[p] <- model@model$cross_validation_metrics@metrics$logloss
      h2o.rm( "model" )
      #rm( model )
      gc()
    }   
    
    
    cbind( tuning_params ,tuning_results) [order(tuning_results),]
    saveRDS(tuning_results , file="data/rf_tuning_6.rds")
    
    
    h2o.shutdown(prompt = FALSE)
    
    # selected model
    #depth=50
    # mtry=85
    min_rwos=10
    ###      
    
    
    
    
    
    
    
    
    
    
    
    tuning_params_2 <- expand.grid( max_depth = c(50,60),
                                    mtries= c(40,50) ,
                                    min_rows= c(5,10,15))
    
    tuning_results_2 <- numeric(  nrow(tuning_params_2) )
    
    
    for ( p in 1:nrow(tuning_params_2) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "QuoteConversion_Flag", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 200,
                         max_depth = tuning_params_2[p,1],
                         mtries= tuning_params_2[p,2],
                         min_rows = tuning_params_2[p,3] ,
                         stopping_metric="AUC")
      
      
      tuning_results_2[p] <- model@model$cross_validation_metrics@metrics$AUC
    }   
    
    
    
    cbind( tuning_params_2 ,tuning_results_2) [order(tuning_results_2,decreasing = TRUE),]
    
    ###  
    
    tuning_params_3 <- expand.grid( max_depth = c(30,40),
                                    mtries= c(70,80) ,
                                    min_rows= c(1,3,5)   )
    
    tuning_results_3 <- numeric(  nrow(tuning_params_3) )
    
    
    for ( p in 1:nrow(tuning_params_3) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "QuoteConversion_Flag", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 500,
                         max_depth = tuning_params_3[p,1],
                         mtries=     tuning_params_3[p,2],
                         min_rows =  tuning_params_3[p,3] ,
                         stopping_metric="AUC")
      
      
      tuning_results_3[p] <- model@model$cross_validation_metrics@metrics$AUC
    }   
    
    
    
    cbind( tuning_params_3 ,tuning_results_3) [order(tuning_results_3,decreasing = TRUE),]
    
    ###  
    tuning_params_4 <- expand.grid( max_depth = c(50,60),
                                    mtries= c(70,80) ,
                                    min_rows= c(1,3)   )
    
    tuning_results_4 <- numeric(  nrow(tuning_params_4) )
    
    
    for ( p in 1:nrow(tuning_params_3) ){
      #    p <- 1
      model <- 
        h2o.randomForest(y = "QuoteConversion_Flag", 
                         x = 1:304, 
                         training_frame = train_subset.h2o ,
                         fold_column = "fold" , 
                         ntrees = 500,
                         max_depth = tuning_params_4[p,1],
                         mtries=     tuning_params_4[p,2],
                         min_rows =  tuning_params_4[p,3] ,
                         stopping_metric="AUC")
      
      
      tuning_results_4[p] <- model@model$cross_validation_metrics@metrics$AUC
    }   
    
    
    
    cbind( tuning_params_4 ,tuning_results_4) [order(tuning_results_4,decreasing = TRUE),]
    
    ###  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    final_model <- 
      h2o.randomForest(y = "QuoteConversion_Flag", 
                       x = 1:304, 
                       training_frame = train.h2o ,
                       ntrees = 500,
                       max_depth = final_params[1],
                       mtries= final_params[2],
                       min_rows = final_params[3] )
    
    
    pred_labels <- h2o.predict(model, test.h2o)
    str(pred_labels)
    
    
    
    
    
    
    
    
    
    
    
    grid <- h2o.grid("randomForest", 
                     y = "QuoteConversion_Flag", 
                     x = 1:304, 
                     training_frame = train_subset.h2o[(fold_subset.h2o!=f)>0] , 
                     validation_frame = train_subset.h2o[fold_subset.h2o==f] , 
                     nfolds= 5 ,
                     hyper_params = list(ntrees = 500,
                                         max_depth = c(10,20,30),
                                         mtries= c(10,20,30) ,
                                         min_rows= c(5,15,25),
                                         stopping_metric="AUC" ) 
    )      
    
    summary(grid)
    
    
    
    
    best_model <- grid@model[[1]]
    best_params <- best_model@model$params
    
    
    
    
    model <- 
      h2o.randomForest(y = "QuoteConversion_Flag", 
                       x = 1:304, 
                       training_frame = train.h2o , 
                       ntrees = c(100,250,500),
                       max_depth = c(10,20,30),
                       mtries= c(10,20,30),
                       nfolds= 5 ,
                       stopping_metric="AUC")
    
    pred_labels <- h2o.predict(model, test.h2o)[,1]
    
    
    #################################
    ## Example of tune grid with for loop
    
    ntree = seq(100,500,100)
    balance_class = c(TRUE,FALSE)
    learn_rate = seq(.05,.4,.05)
    
    parameters = list(ntree = c(), balance_class = c(), learn_rate = c(), r2 = c(), min.r2 = c(), max.r2 = c(), acc = c(), min.acc = c(), max.acc = c(), AUC = c(), min.AUC = c(), max.AUC = c())
    n = 1
    
    mooc.hex = as.h2o(localH2O, mooc[,c("enrollment_id","dropped_out_factor",x.names)])
    for (trees in ntree) {
      for (c in balance_class) {
        for (rate in learn_rate) {
          r2.temp = c(NA,NA,NA)
          acc.temp = c(NA,NA,NA)
          auc.temp = c(NA,NA,NA)
          for (i in 1:3) {
            
            mooc.hex.split = h2o.splitFrame(mooc.hex, ratios=.8)   
            train.gbm = h2o.gbm(x = x.names, y = "dropped_out_factor",  training_frame = mooc.hex.split[[1]],
                                validation_frame = mooc.hex.split[[2]], ntrees = trees, balance_classes = c, learn_rate = rate)
            r2.temp[i] = train.gbm@model$validation_metrics@metrics$r2
            acc.temp[i] = train.gbm@model$validation_metrics@metrics$max_criteria_and_metric_scores[4,3]
            auc.temp[i] = train.gbm@model$validation_metrics@metrics$AUC
          }
          parameters$ntree[n] = trees
          parameters$balance_class[n] = c
          parameters$learn_rate[n] = rate
          parameters$r2[n] = mean(r2.temp)
          parameters$min.r2[n] = min(r2.temp)
          parameters$max.r2[n] = max(r2.temp)
          parameters$acc[n] = mean(acc.temp)
          parameters$min.acc[n] = min(acc.temp)
          parameters$max.acc[n] = max(acc.temp)
          parameters$AUC[n] = mean(auc.temp)
          parameters$min.AUC[n] = min(auc.temp)
          parameters$max.AUC[n] = max(auc.temp)
          n = n+1
        }
      }
    }
    
    
    parameters.df = data.frame(parameters)
    parameters.df[which.max(parameters.df$AUC),]
    
    
    h2o.shutdown(prompt = FALSE)
    