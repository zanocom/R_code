#########################################
#########################################
# NESTED CROSS VALIDATION
# Model Selection
#########################################
#########################################
# The following code shows how to perform 
# model selection with nested cross validation

# see Sebastian Raschka's page for details
# https://sebastianraschka.com/faq/docs/evaluate-a-model.html
#########################################
# CROSS VALIDATION FOLDS
    
# create list for model performance 
    oos_performance <- list()
    
#########################################
# FUNCTION FOR MODEL PERFORMANCE
    
    f1_score <- function( pred , actual  ){
      # pred: factor of predictions
      # actual: factor of positive / negative cases
     
        library(caret)
        precision <- posPredValue(pred, actual, positive="1")
        recall <- sensitivity(pred, actual, positive="1")
        precision <- ifelse( is.na(precision) , 0 , precision)
        recall <- ifelse( is.na(recall) , 0 , recall)
        
        F1 <- ifelse( (precision + recall)==0 , 0 , (2*precision*recall)/(precision+recall) )
        return( F1 )
    }
          
    
    #########################################
    # LOG REG - HYPERPARAMETERS TUNING WITH NESTED CROSS VALIDATION
    # Given lr_dataset we select the best parameters on the inner folds
    # we evaluate the out of sample error on the outer folds
            
    set.seed ( 4446 )
    outer_folds  <- 3
    inner_folds  <- 3
    # grid search
    threshold <- seq( 0.05 , 0.5 , by= 0.025 )
    
    fold_index <- cbind( rep(1:outer_folds,each=inner_folds) , rep(1:inner_folds) )
    idx_outer  <- sample(rep( c(1:outer_folds) , length= nrow(lr_dataset)) , replace= FALSE)
    
    
    
    # for each outer fold there are 3 inner_folds
    idx_inner_list <- list(   )
    set.seed(473)
    for ( o in 1:outer_folds )  # folds
    {
      # outer training fold
      train_fold  <- full_dataset[idx_outer!= o , ]  
      # for each outer fold there are i inner_folds
      idx_inner_list[[o]] <- sample(rep( c(1: inner_folds), length= nrow(train_fold)) , replace= FALSE )
    }
    
    
    # cross validation error matrix
    # fold x parameters
    f1_mat <-  matrix ( 0 , nrow = nrow(fold_index) , ncol =length(threshold)   )  # fold x parameters
    
    
    
    # outer loop
    for ( f in  1:nrow(fold_index) )  
    {
      #f <- 2
      outer_index <- fold_index[f,1]
      inner_index <- fold_index[f,2]
      start.time <- Sys.time()
      
      cat("Start outer fold", outer_index, inner_index , "\n")
      
      cv_fold_outcome  <- lr_dataset[idx_outer!=outer_index ,][idx_inner_list[[outer_index]]==inner_index,]
      

      # fit 
      fit.logistic <- glm( lapsed_next_period~ ., data= lr_dataset[idx_outer!=outer_index ,][idx_inner_list[[outer_index]] != inner_index,]  , family= 'binomial')
      predicted_response_test_numeric <- predict(fit.logistic , newdata=lr_dataset[idx_outer!=outer_index ,  ][idx_inner_list[[outer_index]] == inner_index,] , type = "response"   )
      significant_variables[[f]] <- coef(summary(fit.logistic))[,4]
      
      # parameters loop
      for ( p in 1:length(threshold) )   # parameters
      {
        predicted_response_test <- factor((predicted_response_test_numeric>= threshold[p] )*1)
        f1_mat[f,p]  <- f1_score( predicted_response_test, cv_fold_outcome$lapsed_next_period)  
      }
      rm (  cv_fold_outcome ) ; gc() #, train_fold_outcome
      
      cat("End outer fold", outer_index , "\n")
      end.time <- Sys.time()
      time.taken <- difftime( end.time , start.time,  units="secs")
      cat( round( time.taken / 60 , 1 ) , "minutes taken" ,  "\n" )
      # Fine  out fold  
    }
    
    
    
    
    #########################################
    # FEATURE SELECTION

   significant_variables_df <-  data.frame(t(sapply(significant_variables,c)))
   
   #select variables withi significant pvalues in more than3 folds
   which(apply(significant_variables_df,2, function(x) sum(x<=0.05)  )>=3)
    
   summary( fit.logistic )
    
    #########################################
    # HYPERPARAMETERS SELECTION
    # Best parameters have high F1-score and low complexity
    
    f1_cv <- apply( f1_mat , 2 , mean)
    threshold <- cbind(threshold , f1_cv )
    plot(  f1_cv , pch=19 , col="red") 
    best_f1 <- threshold[threshold[,2]==max(f1_cv),]
    
    
    #########################################
    # COMPUTE OUTER CV PERFORMANCE 
    # USING SELECTED PARAMETERS
    # outer loop
    for ( outer_index in 1:outer_folds )
    {
      start.time <- Sys.time()
      cat("Inizio outer fold", outer_index, "\n")
      
      fit.logistic <- glm( lapsed_next_period~ ., data= lr_dataset[idx_outer!=outer_index,]  , family= 'binomial')
      predicted_response_test_numeric[idx_outer == outer_index] <- predict(fit.logistic , newdata=lr_dataset[idx_outer==outer_index ,] , type = "response"   )
      
      # end outer fold
    }
    
    
    predicted_response_test <- factor( (predicted_response_test_numeric>=best_f1[1] )*1)
    
    # out of sample performance
    oos_performance[["LR"]] <- 
      data.frame( top_10_lift= TopDecileLift( predicted_response_test_numeric , lapsed_next_period_numeric ) )
    
    
    
# Output Representation
    
    output_df <- data.frame( customer_id= full_dataset$customer_id , 
                             pred_prob= predicted_response_test_numeric ,
                             pred_binary= predicted_response_test ,  
                             actual_binary= lr_dataset$lapsed_next_period ,
                             actual_num = lapsed_next_period_numeric )
    
    # confusion matrix
    table(output_df[,c("pred_binary","actual_binary")])
    
    
    
         
#########################################
# RANDOM FOREST    
# HYPERPARAMETERS TUNING WITH NESTED CROSS VALIDATION
# we select the best parameters on hte inner folds
# we evaluate the out of sample errore on the outer folds
    
    set.seed ( 444 )
    outer_folds  <- 3
    inner_folds  <- 3
    
    fold_index <- cbind( rep(1:outer_folds,each=inner_folds) , rep(1:inner_folds) )
    idx_outer  <- sample(rep( c(1:outer_folds) , length= nrow(full_dataset)) , replace= FALSE)
   
    # for each outer fold there are 3 inner_folds
    idx_inner_list <- list(   )
    set.seed(434)
    for ( o in 1:outer_folds )  # folds
    {
      # outer training fold
      train_fold  <- full_dataset[idx_outer!= o , ]  
      # for each outer fold there are i inner_folds
      idx_inner_list[[o]] <- sample(rep( c(1: inner_folds), length= nrow(train_fold)) , replace= FALSE )
    }
    
    # grid search
      # mtry <- seq( 3 , 23 , by= 3 )
      # node <- seq( 6 , 60 , by=6 )
     mtry <- seq( 18 , 22  )
      node <- seq( 6 , 14  )
     # mtry <- seq( 16 , 23 )
     # node <- seq( 6 , 12  )
    parameters <- expand.grid ( mtry  , node )
    colnames ( parameters ) <- c( "mtry" , "node")
 
      
  # cross validation error matrix
    # fold x parameters
    f1_mat <-  matrix ( 0 , nrow = nrow(fold_index) , ncol = nrow(parameters)  )  # fold x parameters
    # for complexity estimate (fold x parameters)
    complexity_mat <-  matrix ( 0 , nrow = nrow(fold_index) , ncol = nrow(parameters)  )  
    
    
    
    set.seed(423)
    # outer loop
    for ( f in  1:nrow(fold_index) )  
    {
      #f <- 2
      outer_index <- fold_index[f,1]
      inner_index <- fold_index[f,2]
      start.time <- Sys.time()
      
      cat("Start outer fold", outer_index, inner_index , "\n")
 
      # variabile da usare dopo il fit per calcolare la performance
      #train_fold_outcome  <- train_set[ idx_outer != outer_index ,c("tfa_out_stima","share_of_wallet","total_asset")][idx_inner_list[[outer_index]] != inner_index,] 
      cv_fold_outcome  <- full_dataset[idx_outer!=outer_index ,-1][idx_inner_list[[outer_index]]==inner_index,]
      
      
      # parameters loop
      for ( p in 1:nrow(parameters) )   # parameters
      {
        #p <- 1
        rf.fit <- randomForest(
                  x=     full_dataset[idx_outer!=outer_index , valid_cols  ][idx_inner_list[[outer_index]] != inner_index,] ,   #escludo flag_feedback_pfa , tfa_out_stima , share_of_wallet
                  y=     full_dataset$lapsed_next_period[idx_outer!= outer_index ][idx_inner_list[[outer_index]] != inner_index] , # variabile risposta share_of_wallet
                  xtest= full_dataset[idx_outer!=outer_index , valid_cols ][idx_inner_list[[outer_index]] == inner_index,] , #calcolo output solo per clienti con feedback
                  ytest= full_dataset$lapsed_next_period[idx_outer!= outer_index ][idx_inner_list[[outer_index]]==inner_index]  ,
                  ntree= 100 ,  
                  mtry=  parameters[p,1] , 
                  nodesize = parameters[p,2], 
                  importance= FALSE, 
                  localImp= FALSE, 
                  nPerm= 1,
                  keep.forest= FALSE, 
                  corr.bias= FALSE,
                  keep.inbag= FALSE ) 
              
        # ogni processore fornisce una risposta
        # trasformo in matrice e calcolo la somma per ogni elemento
        predicted_lapsed <- rf.fit$test$predicted
        
       #  table( predicted_lapsed, cv_fold_outcome$lapsed_next_period )
       #  precision <- posPredValue(predicted_lapsed, cv_fold_outcome$lapsed_next_period ,positive="1")
       #  recall <- sensitivity(predicted_lapsed, cv_fold_outcome$lapsed_next_period ,positive="1")
        # F1 <- (2 * precision * recall) / (precision + recall)
        #is.na(precision)
         
        f1_mat[f,p]  <- f1_score( predicted_lapsed, cv_fold_outcome$lapsed_next_period)  
        complexity_mat[f,p] <-  parameters[p,1]*(nrow(full_dataset[idx_outer!=outer_index,][idx_inner_list[[outer_index]] != inner_index,] )/parameters[p,2])
        rm ( rf.fit  , predicted_lapsed   ) ; gc()# remove i-th level objects, train_predicted_tfa
        # fine fold parameter
      }
      rm (  cv_fold_outcome ) ; gc() #, train_fold_outcome
      
      cat("End outer fold", outer_index , "\n")
      end.time <- Sys.time()
      time.taken <- difftime( end.time , start.time,  units="secs")
      cat( round( time.taken / 60 , 1 ) , "minutes taken" ,  "\n" )
      # Fine  out fold  
    }

    
    
#########################################
# HYPERPARAMETERS SELECTION
# Best parameters have high F1-score and low complexity
    f1_cv <- apply( f1_mat , 2 , mean)
    complexity <- apply( complexity_mat , 2 , mean)
    parameters <- cbind(parameters , f1_cv,  complexity)
    plot( complexity , f1_cv , pch=19 , col="red") 
  
    best_f1 <- parameters[parameters$f1_cv==max(parameters$f1_cv),]
    best_params <- best_f1[which.min(best_f1$complexity),]  
    
    
# Learning Curve
    rf.fit <- randomForest(
      x=     full_dataset[ , valid_cols ] ,   
      y=     full_dataset$lapsed_next_period, 
      ntree= 200 ,  
      mtry=  best_params$mtry , 
      nodesize = best_params$node , 
      importance= TRUE, 
      localImp= FALSE, 
      nPerm= 1,
      keep.forest= FALSE, 
      corr.bias= FALSE,
      keep.inbag= FALSE ) 
    
    # Error reduction
    plot (  rf.fit )
    
    
    # Variables Importance
    rf_importante_variables <- 
    varImpPlot( rf.fit ,
                sort = T,
                main="Variable Importance",
                sub="Decrease in node impurities",
                n.var= 10 ,
                type= 2)

                
                
    #https://gist.github.com/ramhiser/6dec3067f087627a7a85
    var_importance <- data_frame(variable= colnames(full_dataset[ , valid_cols ]) ,
                                 importance= as.vector(importance(rf.fit , type=2)) )
    var_importance <- arrange(var_importance, desc(importance))
    var_importance$variable <- factor(var_importance$variable, levels=var_importance$variable)
    
    p <- ggplot(var_importance[1:10,], aes(x=variable, weight=importance, fill=variable))
    p <- p + geom_bar() + ggtitle("Variable Importance Random Forest")
    p <- p + xlab("Variable") + ylab("Mean Decrease\nin node impurities")
    p <- p + scale_fill_discrete(name="Variable Name")
    p + theme(axis.text.x=element_text(size=10, angle=90 , hjust=1 , vjust=0),
              axis.text.y=element_text(size=9),
              axis.title=element_text(size=10),
              plot.title=element_text(size=12))+guides(fill=FALSE)

    
    
#########################################
# COMPUTE OUTER CV PERFORMANCE 
# USING SELECTED PARAMETERS
    rf_test_pred <-  rep (0 , length = nrow(full_dataset) )  
    rf_test_probs <-  rep (0 , length = nrow(full_dataset) ) 
    set.seed(843)
  #  outer_folds  <- 3  
  #  idx_outer <- sample( rep( c(1: outer_folds) , length= nrow(full_dataset)) , replace= FALSE)
  
    # outer loop
    for ( outer_index in 1:outer_folds )
    {
      start.time <- Sys.time()
      cat("Inizio outer fold", outer_index, "\n")
     
      rf.fit <- 
        randomForest(
          x=     full_dataset[idx_outer != outer_index ,valid_cols ], 
          y=     full_dataset$lapsed_next_period[ idx_outer != outer_index ], 
          xtest= full_dataset[idx_outer == outer_index , valid_cols] , 
          ytest= full_dataset$lapsed_next_period[ idx_outer == outer_index ] ,
          ntree= 50 ,  
          mtry=  best_params$mtry , 
          nodesize = best_params$node ,
          importance= FALSE, 
          localImp= FALSE, 
          nPerm= 1,
          norm.votes= TRUE, 
          keep.forest= FALSE, 
          corr.bias= FALSE,
          keep.inbag= FALSE ) 
      
      # inserisco la previsione test per fold outer
      rf_test_pred[idx_outer == outer_index] <- rf.fit$test$predicted 
      rf_test_probs[idx_outer == outer_index] <- rf.fit$test$votes[,2] 
      rm ( rf.fit  ) ; gc()  # remove i-th fold glm.fit
      # end outer fold
    }
    
    
    # out of sample performance
    oos_performance[["RF"]] <- 
      data.frame( top_10_lift= TopDecileLift( rf_test_probs , lapsed_next_period_numeric ) )
    

    
    
#########################################
## DATASET PROCESSING FOR XGBOOST
    
    outcome_knn <- full_dataset$lapsed_next_period
    excluded_cols_knn <- c("customer_id","lapsed_next_period","flag_multi_order","flag_returned_order","flag_past_returned_order","flag_latest_returned_order" )
    full_dataset_knn <-  full_dataset %>% select( -which(colnames(full_dataset) %in% excluded_cols_knn) )  
    
    
    dummy_list <- list()
    for( attr in colnames(full_dataset_knn) )
    {
      if( is.factor(full_dataset_knn[[attr]]) )
      {
        pre_df <- data.frame( dummies::dummy(x= full_dataset_knn[[attr]] , sep= "") )
        colnames( pre_df ) <-  paste( as.character(attr) , 1:dim(pre_df)[2] , sep="" )
        dummy_list[[attr]]<-pre_df  
      }
    }
    
    # convert from list of dataframes into  dataframe 
    dummy_cols <- dplyr::bind_cols( dummy_list )
    # remove columns that ends with 1 to avoid redundanne
    dummy_cols <- dummy_cols %>% select(-ends_with("1"))
    
    # remove factor variabiles from original dataset
    factor_vars_index <-   which( unlist( lapply( full_dataset_knn,is.factor) )  )
    full_dataset_knn <- full_dataset_knn %>% select( -factor_vars_index )  
    full_dataset_knn <- cbind(full_dataset_knn , dummy_cols)
    
    
    
    ########################
    # Model XGBOOST
    library ( caret )
    library ( xgboost )
  
    # DMatrix used by xgboost functino
    dtrain <- xgb.DMatrix( data= as.matrix(full_dataset_knn) ,
                           label= lapsed_next_period_numeric )
    
    #xgb.DMatrix.save(dtrain, 'xgb.DMatrix_20171015.data')
 
       #######################  
    # Grid Search
     # there are three types of parameters: General Parameters, Booster Parameters and Task Parameters.
    # General parameters refers to which booster we are using
    # Booster parameters depends on which booster you have chosen
    # Learning Task parameters that decides on the learning scenario
    
  
    # set up the cross-validated hyper-parameter search
    xgb_grid_1 = expand.grid(
      eta = c(1e-4,1e-3,1e-2,1e-1 ), # eta actually shrinks the feature weights
      max_depth = c(2, 4, 6, 8, 10 , 12 ), # maximum depth of a tree
      gamma=0,    # minimum loss reduction required to make a further partition on a leaf node of the tree.
      colsample_bytree=1, # subsample ratio of columns when constructing each tree
      min_child_weight=1 # minimum sum of instance weight(hessian) needed in a child to make a further partition 
    )
    
    
     labels <- getinfo( dtrain, "label") 
    
     
     ##########################################
    # HYPERPARAMETERS TUNING WITH NESTED CV
    set.seed ( 444 )
    # definisco struttura della cross validation
    outer_folds  <- 3
    inner_folds  <- 3
    
    # griglia degli indici per identificater outer e inner fold
    fold_index <- cbind( rep(1:outer_folds , each= inner_folds) , rep(1:inner_folds )  )
    dtrain_rows <- getinfo(dtrain, "nrow")
    
    # selezione casuale delle righe per outer e inner folds
    idx_outer <- sample(  rep( c(1: outer_folds), length= dtrain_rows) , replace= FALSE )
    
    idx_inner_list <- list( )
    set.seed(434)
    for ( o in 1:outer_folds )  # folds
    {
      train_fold  <- slice(dtrain,  which(idx_outer != o) )
      righe_inner <-  getinfo(train_fold, "nrow")
      idx_inner_list[[o]] <- sample(  rep( c(1:inner_folds), length=righe_inner ) , replace= FALSE )
    }
    
   # cv metrics matrix ( fold x parameters  )
    f1_mat <-  matrix (0 , nrow = outer_folds , ncol = nrow(xgb_grid_1)  )  
    # test set predictions (dtrain_rows x parameters)
    test_pred <-  matrix ( 0 , nrow = dtrain_rows , ncol = nrow(xgb_grid_1)  ) 
    
    
    library ( parallel )
    no_cores <- detectCores() 
    
    # outer loop
    for ( f in 1:outer_folds )  
    {
      #f<-1
      start.time <- Sys.time()
      outer_index <- f
      # inner_index <-fold_index[f,2]
      cat("Start outer fold", outer_index , "\n")
      
      train_fold_subset  <-  slice(dtrain,  which(idx_outer != outer_index) ) 
      train_fold_outcome <-  getinfo(train_fold_subset , "label") 
      test_fold_subset   <-  slice(dtrain,  which(idx_outer == outer_index) ) 
      test_fold_outcome  <-  getinfo(test_fold_subset , "label") 
      
      # cross validation index
      # da utilizzare in xgb.cv
      cv_list <- list()
      for ( l in 1:inner_folds )  # folds
          {
            cv_list[[l]] <- which( idx_inner_list[[outer_index]]==l )
          }
      
      
      # xgboost initial params
      nrounds <- 100
      param <- list()
      
      
      for ( p in 1:nrow(xgb_grid_1) ) {
        #p<-1
        # iterate over grid params
        param <- list(max_depth=xgb_grid_1$max_depth[p], 
                      eta=xgb_grid_1$eta[p] , 
                      gamma= xgb_grid_1$gamma[p] ,  
                      colsample_bytree=xgb_grid_1$colsample_bytree[p] , 
                      min_child_weight=xgb_grid_1$min_child_weight[p] )
        
        
        # for each training dataset
        # return predicition of cv fold
        cv_model <- xgb.cv(param, 
                           train_fold_subset, 
                           folds= cv_list , # lista custom di indici per elementi di cross validation
                           nround= nrounds , 
                           showsd = TRUE, 
                           objective = "binary:logistic", 
                           nthread=no_cores , 
                           verbose = FALSE, 
                           metrics= "error" ,
                           prediction= TRUE)
        
        
        # quindi calcoliamo l'evaluation error
        #cv_err_mat[f,p] <- evalerror( cv_model$pred , train_fold_subset ) 
       f1_mat[f,p]  <-  f1_score( factor((cv_model$pred>=0.5)*1),factor(train_fold_outcome) )  
 
        # fine loop parameter
      }
      cat("End outer fold", outer_index , "\n")
      end.time <- Sys.time()
      time.taken <- difftime ( end.time , start.time,  units="secs")
      cat ( round( time.taken / 60 , 1 ) , "minutes taken" ,  "\n" )
      # fine loop outer
    }
    
    
    
    
    ### MODEL SELECTION 
    f1_cv_xgb <- apply(f1_mat , 2, mean )
   
     
    plot( f1_cv_xgb , type="n"   )
    points( f1_cv_xgb , col="red" ,pch=19 )

    
    # best params
    xgb_grid_1[which.max(f1_cv_xgb),]
    
    #######################################################
    # FINE TUNING
    
     # set up the cross-validated hyper-parameter search

    set.seed(555)
    xgb_grid_2 = expand.grid(
      eta =  c(1e-5,1.5e-5,1e-4,1.5e-4,1e-3 ), # eta actually shrinks the feature weights
      max_depth = c( 8,10,12,14,16  ), # maximum depth of a tree
      gamma=0,    # minimum loss reduction required to make a further partition on a leaf node of the tree.
      colsample_bytree=1, # subsample ratio of columns when constructing each tree
      min_child_weight=1 # minimum sum of instance weight(hessian) needed in a child to make a further partition 
    )
    
    labels <- getinfo( dtrain, "label") 
    
    
    # cv metrics matrix ( fold x parameters  )
    f1_mat <-  matrix (0 , nrow = outer_folds , ncol = nrow(xgb_grid_2)  )  
   
      # outer loop
    for ( f in 1:outer_folds )  
    {
      #f<-1
      start.time <- Sys.time()
      outer_index <- f
      # inner_index <-fold_index[f,2]
      cat("Start outer fold", outer_index , "\n")
      
      train_fold_subset  <-  slice(dtrain,  which(idx_outer != outer_index) ) 
      train_fold_outcome <-  getinfo(train_fold_subset , "label") 
      test_fold_subset   <-  slice(dtrain,  which(idx_outer == outer_index) ) 
      test_fold_outcome  <-  getinfo(test_fold_subset , "label") 
      
      # cross validation index
      # da utilizzare in xgb.cv
      cv_list <- list()
      for ( l in 1:inner_folds )  # folds
      {
        cv_list[[l]] <- which( idx_inner_list[[outer_index]]==l )
      }
      
      
      # xgboost initial params
      nrounds <- 100
      param <- list()
      
      
      for ( p in 1:nrow(xgb_grid_2) ) {
        #p<-1
        # iterate over grid params
        param <- list(max_depth=xgb_grid_2$max_depth[p], 
                      eta=xgb_grid_2$eta[p] , 
                      gamma= xgb_grid_2$gamma[p] ,  
                      colsample_bytree=xgb_grid_2$colsample_bytree[p] , 
                      min_child_weight=xgb_grid_2$min_child_weight[p] )
        
        
        # for each training dataset
        # return predicition of cv fold
        cv_model <- xgb.cv(param, 
                           train_fold_subset, 
                           folds= cv_list , # lista custom di indici per elementi di cross validation
                           nround= nrounds , 
                           showsd = TRUE, 
                           objective = "binary:logistic", 
                           nthread=no_cores , 
                           verbose = FALSE, 
                           metrics= "error" ,
                           prediction= TRUE)
        
        
        # quindi calcoliamo l'evaluation error
        #cv_err_mat[f,p] <- evalerror( cv_model$pred , train_fold_subset ) 
        f1_mat[f,p]  <-  f1_score( factor((cv_model$pred>=0.5)*1),factor(train_fold_outcome) )  
        
        # fine loop parameter
      }
      cat("End outer fold", outer_index , "\n")
      end.time <- Sys.time()
      time.taken <- difftime ( end.time , start.time,  units="secs")
      cat ( round( time.taken / 60 , 1 ) , "minutes taken" ,  "\n" )
      # fine loop outer
    }
    
    
    ### MODEL SELECTION 
    f1_cv_xgb <- apply(f1_mat , 2, mean )
    
    plot( f1_cv_xgb , type="n"   )
    points( f1_cv_xgb , col="red" ,pch=19 )
    
    
    # best params
    xgb_grid_2[which.max(f1_cv_xgb),]
    xgb_grid_2$f1_cv_xgb <- f1_cv_xgb
    
    
    ##########
    set.seed(555)
    xgb_grid_3 = expand.grid(
      eta =  c(1e-5,1.25e-5,1.5e-5,1.75e-5,1e-4 ), # eta actually shrinks the feature weights
      max_depth = c( 14,15,16,17,18 ), # maximum depth of a tree
      gamma=0,    # minimum loss reduction required to make a further partition on a leaf node of the tree.
      colsample_bytree=1, # subsample ratio of columns when constructing each tree
      min_child_weight=1 # minimum sum of instance weight(hessian) needed in a child to make a further partition 
    )
    
    labels <- getinfo( dtrain, "label") 
    
    
    # cv metrics matrix ( fold x parameters  )
    f1_mat <-  matrix (0 , nrow = outer_folds , ncol = nrow(xgb_grid_3)  )  
    
    # outer loop
    for ( f in 1:outer_folds )  
    {
      #f<-1
      start.time <- Sys.time()
      outer_index <- f
      # inner_index <-fold_index[f,2]
      cat("Start outer fold", outer_index , "\n")
      
      train_fold_subset  <-  slice(dtrain,  which(idx_outer != outer_index) ) 
      train_fold_outcome <-  getinfo(train_fold_subset , "label") 
      test_fold_subset   <-  slice(dtrain,  which(idx_outer == outer_index) ) 
      test_fold_outcome  <-  getinfo(test_fold_subset , "label") 
      
      # cross validation index
      # da utilizzare in xgb.cv
      cv_list <- list()
      for ( l in 1:inner_folds )  # folds
      {
        cv_list[[l]] <- which( idx_inner_list[[outer_index]]==l )
      }
      
      
      # xgboost initial params
      nrounds <- 100
      param <- list()
      
      
      for ( p in 1:nrow(xgb_grid_3) ) {
        #p<-1
        # iterate over grid params
        param <- list(max_depth=xgb_grid_3$max_depth[p], 
                      eta=xgb_grid_3$eta[p] , 
                      gamma= xgb_grid_3$gamma[p] ,  
                      colsample_bytree=xgb_grid_3$colsample_bytree[p] , 
                      min_child_weight=xgb_grid_3$min_child_weight[p] )
        
        
        # for each training dataset
        # return predicition of cv fold
        cv_model <- xgb.cv(param, 
                           train_fold_subset, 
                           folds= cv_list , # lista custom di indici per elementi di cross validation
                           nround= nrounds , 
                           showsd = TRUE, 
                           objective = "binary:logistic", 
                           nthread=no_cores , 
                           verbose = FALSE, 
                           metrics= "error" ,
                           prediction= TRUE)
        
        
        # quindi calcoliamo l'evaluation error
        #cv_err_mat[f,p] <- evalerror( cv_model$pred , train_fold_subset ) 
        f1_mat[f,p]  <-  f1_score( factor((cv_model$pred>=0.5)*1),factor(train_fold_outcome) )  
        
        # fine loop parameter
      }
      cat("End outer fold", outer_index , "\n")
      end.time <- Sys.time()
      time.taken <- difftime ( end.time , start.time,  units="secs")
      cat ( round( time.taken / 60 , 1 ) , "minutes taken" ,  "\n" )
      # fine loop outer
    }
    
    
    ### MODEL SELECTION 
    f1_cv_xgb <- apply(f1_mat , 2, mean )
    plot( f1_cv_xgb , type="n"   )
    points( f1_cv_xgb , col="red" ,pch=19 )
    
    
    # best params
    xgb_grid_3[which.max(f1_cv_xgb),]
    xgb_grid_3$f1_cv_xgb <- f1_cv_xgb
    
     
    best_params <- xgb_grid_3[which.max(f1_cv_xgb),]
    
 
       
     
    ############################################
    ### SELECT BEST ITERATIONS
    early_stop <- 10
    nfold_cv <- 3
    set.seed( 999 )
    param <- list(max_depth= best_params$max_depth, 
                  eta=       best_params$eta , 
                  gamma=     best_params$gamma ,  
                  colsample_bytree= best_params$colsample_bytree , 
                  min_child_weight= best_params$min_child_weight )
    
    cv_model <- xgb.cv(param, 
               train_fold_subset, 
               folds= cv_list , # lista custom di indici per elementi di cross validation
               nround= 10000 , 
               showsd = TRUE, 
               objective = "binary:logistic", 
               nthread= 4 , 
               verbose = FALSE, 
               metrics= "error" ,
               prediction= TRUE)
    
    plot( cv_model$dt[, test.error.mean]  , type="l" )
    
    # best iterations
    # print( cv_model )
    best_nrounds <- round( which.min(cv_model$dt[,test.error.mean]) / (1-1/nfold_cv) , 0) 
    # 20
    
    ###########################################
    # OUT OF SAMPLE PREDICTION
    test_pred <-  rep ( 0 , length = length(labels)  ) 
    
    # outer loop
    for ( f in 1:outer_folds )  
    {
      #f <- 1
      start.time <- Sys.time()
      outer_index <-f
      cat("Start outer fold", outer_index , "\n")
      
      train_fold_subset  <-  slice(dtrain,  which(idx_outer != outer_index) ) 
      train_fold_outcome <-  getinfo(train_fold_subset , "label") 
      
      test_fold_subset   <-  slice(dtrain,  which(idx_outer == outer_index) ) 
      test_fold_outcome  <-  getinfo(test_fold_subset , "label") 
      
      
      # per ogni fold calcolo la previsione sul test fold
      cat("Start test fold", outer_index , "\n")
      
      test_model <- xgb.train(param, 
                              train_fold_subset, 
                              nround= best_nrounds , 
                              showsd = TRUE, 
                              objective = "binary:logistic", 
                              nthread= 4 , 
                              verbose = FALSE, 
                              metrics= "error",
                              maximize = FALSE ,
                              prediction= TRUE)
         
      test_pred[which(idx_outer == outer_index)] <- predict(test_model, test_fold_subset)
      # fine loop outer
    }
    
     
    # out of sample performance
    oos_performance[["XGB"]] <- 
      data.frame(f1_score= f1_score( factor((test_pred>=0.5)*1,labels=c(0,1)) , full_dataset$lapsed_next_period ) ,
                 top_10_lift= TopDecileLift( test_pred , lapsed_next_period_numeric ) )
    

    
#########################################
#########################################