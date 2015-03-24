################################################################################
##################### Kaggle Criteo Challenge ##################################
################################################################################
## clear data
rm(list = ls())
cat("\014")

cols_to_keep<-c(6,9,14,17,20,22,23)
ntrain<-1000000
ntest<-6042136
blockCount<-10000
trainFile<-"/home/harini/Kaggle_data/Criteo/train_reduced.csv"
trainSQLFile<-"/home/harini/Kaggle_data/Criteo/train_db_sql"
#trainFile<-"../../../../Kaggle_CriteoChallenge/train.csv"
testFile<-"/home/harini/Kaggle_data/Criteo/test.csv"
testSQLFile<-"/home/harini/Kaggle_data/Criteo/test_db_sql"
#testFile<-"../../../../Kaggle_CriteoChallenge/temp.csv"
outFile<-"criteo_submission_rf_07252014.csv"

##load packages
library(randomForest) 
library(sqldf)

Mode <- function(x) {
  ux <- unique(x)
  ux <- setdiff(ux,NA)
  max_val = ux[which.max(tabulate(match(x, ux)))]
  x[is.na(x)]<-max_val
  list(mode_val=max_val,unique_val=ux,data=x)
}

process_data <- function(data){
  keeps <- c(paste("I",sep="",c(1:13)),paste("C", sep="", cols_to_keep))
  data <- data[,(names(data) %in% keeps)]
  int_cols <- c(paste("I",sep="",c(1:13)))
  for (i in int_cols){
    data[,i] <-  as.integer(data[,i])  
  }
  factor_cols <- c(paste("C", sep="", cols_to_keep))
  for (i in factor_cols){
    data[,i] <-  as.factor(data[,i])  
  }  
  temp<-sapply(data, Mode)
  max_val <- temp[1,]
  unique_val <- temp[2,]
  data[,1:ncol(data)]<- temp[3,] 
  list(data=data,max_val=max_val,unique_val=unique_val)
}

process_data_test <- function(data,max_val,unique_val){
  keeps <- c(paste("I",sep="",c(1:13)),paste("C", sep="", cols_to_keep))
  data <- data[,(names(data) %in% keeps)]
  int_cols <- c(paste("I",sep="",c(1:13)))
  for (i in int_cols){
    data[,i] <-  as.integer(data[,i])  
  }
  factor_cols <- c(paste("C", sep="", cols_to_keep))
  for (i in factor_cols){
    data[,i] <-  as.factor(data[,i])  
  } 
  for (i in colnames(data)){
    if(is.factor(data[,i])){
      data[,i]<-factor(data[,i],levels=unique_val[[i]])
    }
    data[is.na(data[i]),i]<-max_val[[i]]
    id <- which(!(data[,i] %in% unique_val[[i]]))
    if(!(length(id)==0)){data[id,i]<- max_val[[i]]}
  }
  data
}

## read train data base if doesnt exist
if (file.exists(trainSQLFile)) {
  sprintf("File exists\n")
}else{
  sqldf(paste("attach '",trainSQLFile,"' as new",sep=""))
  read.csv.sql(trainFile, sql = "create table train as select * from file", dbname = trainSQLFile)
}

#read some training data
s <- sprintf("select * from main.train limit %d, %d", 0, ntrain) 
data<-sqldf(s, dbname = trainSQLFile)
y <- as.factor(data[,"Label"])
temp<-process_data(data)
max_val <- temp$max_val
unique_val <- temp$unique_val
data<- temp$data
rm(temp)

## run Random Forest
rf<-randomForest(x=data,y=y,ntree=100,maxnodes=100)
rm(data)

## read test database if it doesnt exist
if (file.exists(testSQLFile)) {
  sprintf("File exists\n")
}else{
  sqldf(paste("attach '",testSQLFile,"' as new",sep=""))
  read.csv.sql(testFile, sql = "create table main.test as select * from file", dbname = testSQLFile)
}

#read test data in chunks and predict
item_ids <- c()
predictions <- c()
for(i in seq(0, ntest, blockCount)) { 
  s <- sprintf("select * from main.test limit %d, %d", i, blockCount) 
  data<-sqldf(s, dbname = testSQLFile)
  item_ids<-rbind(item_ids,data["Id"])
  data <- process_data_test(data,max_val,unique_val)
  
  temp<-predict(rf, newdata=data, type="prob",norm.votes=TRUE, predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  temp <- temp + 0.01 # to avoid inf log loss
  predictions<-cbind(predictions,t(temp[,2]))
  
  #write output into submission file
  outdata<-data.frame(item_ids,t(predictions))
  colnames(outdata)<-c("Id","Predicted")
  if(i==0){
    write.table(outdata, file = outFile,sep=",", row.names=FALSE,col.names=TRUE,append = TRUE)
  }else{
    write.table(outdata, file = outFile,sep=",", row.names=FALSE,col.names=FALSE,append = TRUE)
  }
  temp<-c()
  item_ids <- c()
  predictions <- c()
  print(i)
} 


