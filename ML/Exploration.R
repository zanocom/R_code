

#####################################################################
#####################################################################
# ALLSTATE INSURANCE KAGGLE
# 2014/02

#####################################################################
#####################################################################

rm ( list = ls ( ))

train_data <- read.table ( "train.csv" , sep="," , header=T)

head ( train_data )
summary ( train_data )


library ( dplyr )
library ( ggplot2 )

train_df <- tbl_df ( train_data ) 
head ( train_df )

summary ( train_df )

# count rows per customer

rows_per_customer <-
train_df %.%
  group_by ( customer_ID )  %.%
  summarise ( righe = n(  )  , min_day=min(day)  , max_day=max(day)    ) %.%
  mutate ( duration = max_day- min_day)
  

# Distribuzione delle righe per cliente: variano da 3 a 13, moda=7
ggplot( rows_per_customer , aes(x=righe) ) +
  geom_histogram()

# Distribuzione della duration per cliente: varia da 0 a 5, moda=0
ggplot( rows_per_customer , aes(x=duration) ) +
  geom_histogram()

ggplot( rows_per_customer , aes( x=righe ) ) +
  geom_histogram() +
  facet_grid(   ~ duration , scales="free_y")



filter (  train_df , customer_ID <= 10000012 )

summary ( train_df$car_value )
summary ( train_df$car_age )
plot ( car_age , car_value , data = train_df    )

ggplot( train_df , aes( x=car_age , y=as.numeric(car_value) ) ) +
  geom_point() 


# Pretrattamento del dataset per correggere il tipo di variabile
str ( train_df  )  # 665249 x 25
summary ( train_df  )

train_df$record_type <- factor ( train_df$record_type , labels =c("Proposal" , "Buy")  )
train_df$location <- factor ( train_df$location  )
train_df$state <- factor ( train_df$state  )
train_df$homeowner <- factor ( train_df$homeowner , labels=c("No","Yes")  )
train_df$married_couple <- factor ( train_df$married_couple , labels=c("No","Yes")  )

train_df$A <- factor ( train_df$A , ordered=T  )
train_df$B <- factor ( train_df$B , ordered=T  )
train_df$C <- factor ( train_df$C , ordered=T  )
train_df$D <- factor ( train_df$D , ordered=T  )
train_df$E <- factor ( train_df$E , ordered=T  )
train_df$F <- factor ( train_df$F , ordered=T  )
train_df$G <- factor ( train_df$G , ordered=T  )


train_df$C_previous[is.na(train_df$C_previous)] <- 0
train_df$duration_previous[is.na(train_df$duration_previous)] <- 20
train_df$C_previous <- factor ( train_df$C_previous , ordered=T  )

# Costruisco un dataset con dati:
# - della transazione output
# - della prima transazione
# - della transazione precedente


# POi creo un dataset train con campionamento ripetuto delle righe del primo dataset


acquisti <- filter (  train_df , record_type== "Buy"  )  # 97009x25



# Salvataggio dataset trattati per applicazione modelli
save(train_df, file = "train_df.RData")
save(acquisti, file = "acquisti.RData")



str ( acquisti )
pairs ( acquisti[,c(18:24)]   )

# Modello basato su dati del Cliente
# Dobbiamo costruire 7 modelli, uno per ciascun prodotto
# Per ogni modello dobbiamo applicare una multiclass classification

# Logistic Regression
# Step Function
# Polynomial Logistic Regression
# Random Forest
# SVM


###################################################
# Validation Set Approach
set.seed ( 1 )
train <- sample ( x = nrow ( acquisti )  ,  size = ( nrow (acquisti)- 24252 )  )
length ( train )
save (   train , file = "train_sample.RData")
test <- c(1:97009)[-train]
save (   test , file = "test_sample.RData")
###################################################
# Logistic Regression Stepwise 
str ( acquisti )
names ( acquisti)
summary ( acquisti )


# Non includo risk_factor negli input perché ha troppe righe mancanti
glm.fit <- glm ( I(A==0) ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                   married_couple+C_previous+duration_previous+cost , family = "binomial" , data = acquisti , subset = train )
summary ( glm.fit )

length ( train )
log (  72757 )

library ( MASS )
glm.fit_step <- stepAIC ( glm.fit , direction = "both"  )
glm.fit_step_bic <- stepAIC ( glm.fit , direction = "both"  , k=11.19)

# Stepwise forward logistic regression
fit1 <- glm ( I(A==0) ~ 1 ,family = "binomial" , data = acquisti , subset = train    )
search_f <- step (fit1  , scope=list(lower= ~ 1, upper= ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                                       married_couple+C_previous+duration_previous) ,  direction = "forward"   )
summary ( search_f )
search_f$anova

# Stepwise forward/backward logistic regression
search_b <- step (fit1  , scope=list(lower= ~ 1, upper= ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                                       married_couple+C_previous+duration_previous) ,  direction = "both"   )
summary ( search_b )
search_b$anova


# I modelli annidati di forward e both stepwise sono identici
# verifichiamo il comportamento di training error vs test error
# per i modelli di diversa complessità


fit_0 <- glm ( I(A==0) ~ 1 ,family = "binomial" , data = acquisti , subset = train    )
fit_1 <- update ( fit_0 , . ~ . + car_age )
fit_2 <- update ( fit_1 , . ~ . + car_value )
fit_3 <- update ( fit_2 , . ~ . + C_previous )
fit_4 <- update ( fit_3 , . ~ . + age_youngest)
fit_5 <- update ( fit_4 , . ~ . + duration_previous)
fit_6 <- update ( fit_5 , . ~ . + homeowner)
fit_7 <- update ( fit_6 , . ~ . + shopping_pt)
fit_8 <- update ( fit_7 , . ~ . + age_oldest)

fit_0_train  <- predict ( fit_0 , type = "response"  )
fit_1_train  <- predict ( fit_1 , type = "response"  )
fit_2_train  <- predict ( fit_2 , type = "response"  )
fit_3_train  <- predict ( fit_3 , type = "response"  )
fit_4_train  <- predict ( fit_4 , type = "response"  )
fit_5_train  <- predict ( fit_5 , type = "response"  )
fit_6_train  <- predict ( fit_6 , type = "response"  )
fit_7_train  <- predict ( fit_7 , type = "response"  )
fit_8_train  <- predict ( fit_8 , type = "response"  )


fit_0_test  <- predict ( fit_0 , type = "response" , newdata = acquisti[-train,]  )
fit_1_test  <- predict ( fit_1 , type = "response" , newdata = acquisti[-train,]  )
fit_2_test  <- predict ( fit_2 , type = "response" , newdata = acquisti[-train,]  )
fit_3_test  <- predict ( fit_3 , type = "response" , newdata = acquisti[-train,]  )
fit_4_test  <- predict ( fit_4 , type = "response" , newdata = acquisti[-train,]  )
fit_5_test  <- predict ( fit_5 , type = "response" , newdata = acquisti[-train,]  )
fit_6_test  <- predict ( fit_6 , type = "response" , newdata = acquisti[-train,]  )
fit_7_test  <- predict ( fit_7 , type = "response" , newdata = acquisti[-train,]  )
fit_8_test  <- predict ( fit_8 , type = "response" , newdata = acquisti[-train,]  )


train_err_0 <-  sum ( (fit_0_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_1 <-  sum ( (fit_1_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_2 <-  sum ( (fit_2_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_3 <-  sum ( (fit_3_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_4 <-  sum ( (fit_4_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_5 <-  sum ( (fit_5_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_6 <-  sum ( (fit_6_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_7 <-  sum ( (fit_7_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      
train_err_8 <-  sum ( (fit_8_train>0.5) != (acquisti$A[train]==0) ) / length(acquisti$A[train])      

test_err_0 <-  sum ( (fit_0_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_1 <-  sum ( (fit_1_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_2 <-  sum ( (fit_2_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_3 <-  sum ( (fit_3_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_4 <-  sum ( (fit_4_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_5 <-  sum ( (fit_5_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_6 <-  sum ( (fit_6_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_7 <-  sum ( (fit_7_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      
test_err_8 <-  sum ( (fit_8_test>0.5) != (acquisti$A[-train]==0) ) / length(acquisti$A[-train])      


plot ( 1:9 , c ( train_err_0 ,train_err_1 , train_err_2 , train_err_3 , train_err_4, 
                 train_err_5 , train_err_6 ,train_err_7 , train_err_8)  , type = "b"     )
points ( 1:9 , c ( test_err_0 ,test_err_1 , test_err_2 , test_err_3 , test_err_4, 
                 test_err_5 , test_err_6 ,test_err_7 , test_err_8)  , type = "b"  , col="red"   )

# Scegliamo il modello logistico fit_4






#####################################################################
# Random Forest
acquisti # dataset da analizzare
str ( acquisti )   #97009 x 25
length (train) # 72757

library ( randomForest  )

mtry_var <- c  ( 1,2,3,4 )
ntree_var <- c ( 50 , 100 , 200 , 300 , 500  )

library ( snowfall )
sfInit(parallel=TRUE, cpus=4, type="SOCK")


rf_train_err <- matrix (0 , ncol=5 , nrow=4)
rf_test_err <-  matrix (0 , ncol=5 , nrow=4)
for ( i in 1:4 ) {
  for ( j in 1:5 ) {
        rf.fit <- randomForest ( A ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                           married_couple+C_previous+duration_previous , type = "classification" ,
                           data = acquisti , subset = train ,
                           mtry = mtry_var[i] , ntree = ntree_var[j] ,
                           xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"A"] 
                           )
        
        rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
        rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1]
        print ( paste (  "Iterazione " , length(mtry_var)*(i-1)+j , " su " , length(mtry_var)*length(ntree_var)   ) ) 
                  }
                }




rf_train_err <- rep ( 0,4)
rf_test_err_b <- rep ( 0,4)
for ( i in 1:4 ) {
  rf.fit <- randomForest ( B ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                             married_couple+C_previous+duration_previous , type = "classification" ,
                           data = acquisti , subset = train ,
                           mtry = mtry_var[i] , ntree = 100 ,
                           xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"B"] )
  
  rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
  rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1]
}



rf_test_err_c <- rep ( 0,4)
for ( i in 1:4 ) {
  rf.fit <- randomForest ( C ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                             married_couple+C_previous+duration_previous , type = "classification" ,
                           data = acquisti , subset = train ,
                           mtry = mtry_var[i] , ntree = 100 ,
                           xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"C"] )
  
  rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
  rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1]
}



rf_test_err_d <- rep ( 0,4)
for ( i in 1:4 ) {
  rf.fit <- randomForest ( D ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                             married_couple+C_previous+duration_previous , type = "classification" ,
                           data = acquisti , subset = train ,
                           mtry = mtry_var[i] , ntree = 100 ,
                           xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"D"] )
  
  rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
  rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1] 
}



rf_test_err_e <- rep ( 0,4)
for ( i in 1:4 ) {
  rf.fit <- randomForest ( E ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                             married_couple+C_previous+duration_previous , type = "classification" ,
                           data = acquisti , subset = train ,
                           mtry = mtry_var[i] , ntree = 100 ,
                           xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"E"] )
  
  rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
  rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1]
  
}


rf_test_err_f <- rep ( 0,4)
for ( i in 1:4 ) {
  rf.fit <- randomForest ( F ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                             married_couple+C_previous+duration_previous , type = "classification" ,
                           data = acquisti , subset = train ,
                           mtry = mtry_var[i] , ntree = 100 ,
                           xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"F"] )
  
  rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
  rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1]
  
}



rf_test_err_g <- rep ( 0,4)
for ( i in 1:4 ) {
  rf.fit <- randomForest ( G ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                             married_couple+C_previous+duration_previous , type = "classification" ,
                           data = acquisti , subset = train ,
                           mtry = mtry_var[i] , ntree = 100 ,
                           xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"G"] )
  
  rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
  rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1]
  
}




######################################################
# BOOSTING

library ( gbm  )


ntree_var <- seq ( 1000 , 3000 , by = 500  )
interaction.depth_var <- c ( 1 , 2 , 3 )
shrinkage_var <- c (  0.001 , 0.01 , 0.1 , 1 )

rf_train_err <- array (0 , ncol=5 , nrow=4)
rf_test_err <-  array (0 , ncol=5 , nrow=4)
for ( i in 1:6 ) {
  for ( j in 1:3 ) {
    for ( k in 1:4 ) {
    rf.fit <- gbm ( A ~ shopping_pt+group_size+homeowner+car_age+car_value+age_oldest+age_youngest+
                               married_couple+C_previous+duration_previous , distribution = "multinomial" ,
                             data = acquisti , subset = train ,
                             n.trees = ntree_var[j] , interaction.depth = , shrinkage= , 
                             #xtest = acquisti[-train,c(2,8:11,13:17)]  , ytest=acquisti[-train,"A"] ,
                    verbose = FALSE
    )
    
    rf_train_err[i,j]  <- rf.fit$err.rate[ ntree_var[j] ,1]
    rf_test_err[i,j] <- rf.fit$test$err.rate[ ntree_var[j] ,1]
                       }
                    }
                  }


#####################################################################
