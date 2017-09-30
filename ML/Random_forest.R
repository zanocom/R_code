####################################################################
#####################################################################
# Parallelized Random Forest
# the code was originally applied to Homesite Contest @ Kaggle
#####################################################################
#####################################################################

rm ( list = ls ( ))

#load( file = "train_df.RData")
#load( file = "acquisti.RData")

###################################################
# Validation Set Approach
load( file = "acquisti.RData")
load( file = "train_sample.RData")
load( file = "test_sample.RData")


library ( parallel )


n_core <-detectCores() # 4
indici_processori <- makeCluster( getOption("cl.cores", n_core) )
vettore_input <- as.vector (11:13) # 18:21  # posizione della variabile y nel dataset di uno specifico ciclo


funzione_rf_idx <- function( k ){
  
  index <- k
  
  # dichiara funzione che effettua random forest
  funzione_rf <- function( idx ){
    #print ( "data loading" )
    load( file = "acquisti.RData")
    ds <- data.frame ( acquisti )[, c(2,8:11,13:17,22:24)]
    ds <- ds [,c(1:10,idx)]
    
    load( file = "train_sample.RData" )
    load( file = "test_sample.RData" )
    train_sample <- train
    test_sample  <- test
    
    #str ( ds )
    #print ( "data loaded" )
    
    mtry_var  <-  c ( 1, 2 , 3 , 4 )
    ntree_var <-  c ( 50 , 100 , 200 , 300 , 500  )
    train_err <-  matrix ( 0 , nrow=length(mtry_var) , ncol=length(ntree_var) )
    test_err  <-  matrix ( 0 , nrow=length(mtry_var) , ncol=length(ntree_var) )
    
   # print ( "y loading" )
    y <- unlist ( ds[,11] )
   # str ( y )
    
   # print ( "y merging" )
    ds$y <- NA
    ds$y <- y
    
   # print ( "y loaded" )
   # str ( ds )
    
    library ( randomForest )
    for ( i in 1: length( mtry_var) ) { 
      for ( j in 1: length( ntree_var) ){
      #  print ("creazione albero")
        fit_rf <-  randomForest ( y ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                                    married_couple+C_previous+duration_previous ,
                                  data = ds , subset = train_sample , 
                                  ntree=ntree_var[j] , mtry=mtry_var[i] ,
                                  xtest = ds [ test_sample , c(1:10) ] , 
                                  ytest = ds$y[ test_sample ]  ,
                                  na.action=na.omit )
      train_err[i,j] <- fit_rf$err.rate[ ntree_var[j] , 1 ]
      test_err[i,j] <- fit_rf$test$err.rate[ ntree_var[j] , 1 ]
      #  print ("fine albero")
      }
    } 
    return ( list ( idx , train_err , test_err ) )
  }
  
  
  funzione_rf (idx=index)  #esegue random forest per ogni valore k
}
 




RNGkind("L'Ecuyer-CMRG")
set.seed(123)
## start M workers
s <- .Random.seed
for (i in 1:4) {
  s <- nextRNGStream(s)
  # send s to worker i as .Random.seed
}




output_rf <- array ( 0 , dim = c ( 4 , 5 , length(vettore_input) ) )  # r x c x h 
output_rf <-  clusterApply( indici_processori , x = vettore_input , fun=funzione_rf_idx   )

# I parametri scelti sono
# 11 2 100 0.2907801
# 12 2 300 0.4471796
# 13 2 100 0.2751938
# 14 2 100 0.3484661
# 15 1 100 0.3469817
# 16 2 500 0.5250701
# 17 2 300 0.5822613


stopCluster( indici_processori )




