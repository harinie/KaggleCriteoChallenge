################################################################################
##################### Kaggle Criteo Challenge ##################################
################################################################################
## clear data
rm(list = ls())
cat("\014")

cols_to_omit<-c(6,9,14,17,20,22,23)
ntrain<-1000000
blockCount<-10000
trainFile<-"/home/harini/Kaggle_data/Criteo/train.csv"
trainSQLFile<-"/home/harini/Kaggle_data/Criteo/train_db_sql"
#trainFile<-"../../../../Kaggle_CriteoChallenge/train.csv"
testFile<-"/home/harini/Kaggle_data/Criteo/test.csv"
testSQLFile<-"/home/harini/Kaggle_data/Criteo/test_db_sql"
#testFile<-"../../../../Kaggle_CriteoChallenge/temp.csv"
outFile<-"criteo_submission_rf_07252014"

##load packages
library(randomForest) 

Mode <- function(x) {
  ux <- unique(x)
  ux <- setdiff(ux,NA)
  max_val = ux[which.max(tabulate(match(x, ux)))]
  x[is.na(x)]<-max_val
  list(mode_val=max_val,unique_val=ux,data=x)
}

process_data <- function(data){
  keeps <- c(paste("I",sep="",c(1:13)),paste("C", sep="", cols_to_omit))
  data <- data[,(names(data) %in% keeps)]
  temp<-sapply(data, Mode)
  max_val <- temp[1,]
  unique_val <- temp[2,]
  data[,1:ncol(data)]<- temp[3,] 
  list(data=data,max_val=max_val,unique_val=unique_val)
}

process_data_test <- function(data,max_val,unique_val){
  keeps <- c(paste("I",sep="",c(1:13)),paste("C", sep="", cols_to_omit))
  data <- data[,(names(data) %in% keeps)]
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

## read data
#data <- read.table(trainFile,header = TRUE,sep=',',fill=TRUE,nrows=100)
data <- read.table(trainFile,header = TRUE,sep=',',fill=TRUE,nrows=ntrain)
y <- as.factor(data.matrix(data["Label"]))
#drops <- c("Id","Label",paste("C", sep="", c(1:5,7:8,10:13,15:16,18:19,21,24,25,26)))
temp<-process_data(data)
max_val <- temp$max_val
unique_val <- temp$unique_val
data<- temp$data
rm(temp)


## run Random Forest
rf<-randomForest(x=data,y=y,ntree=100,maxnodes=100)
rm(data)

## 
library(sqldf)
sqldf("attach test_data_db as new")
read.csv.sql(testFile, sql = "create table main.test as select * from file", dbname = "test_data_db")

# look at first three lines
data<-sqldf("select * from main.test limit 3", dbname = "test_data_db")
sqlcmd <- paste("select * from main.test limit",6,",",11)
data<-sqldf(sqlcmd, row.names = TRUE)


sqldf()
## read test data
count=1
item_ids <- c()
predictions <- c()
header <- read.table(testFile,header = TRUE,sep=',',fill=TRUE,nrows=1)  
header <- colnames(header)
data_size<-blockCount
while(data_size==blockCount){
  data <- read.table(testFile,header = FALSE,sep=',',fill=TRUE,nrows=blockCount,skip=count)  
  data_size<-dim(data)
  data_size<-data_size[1]
  colnames(data)<-header
  item_ids<-rbind(item_ids,data["Id"])
  data <- process_data_test(data,max_val,unique_val)

  temp<-predict(rf, newdata=data, type="prob",norm.votes=TRUE, predict.all=FALSE, proximity=FALSE, nodes=FALSE)
  predictions<-cbind(predictions,t(temp[,2]))
  count = count + blockCount
  temp<-c()
  print(count)
}

outdata<-data.frame(item_ids,t(predictions))
colnames(outdata)<-c("Id","Predicted")

write.table(outdata, file = outFile,  sep = ",",row.names=FALSE,append = FALSE)

