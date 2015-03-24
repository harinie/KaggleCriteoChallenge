#!/usr/bin/python
#Kaggle Challenge: 
#"http://www.kaggle.com/c/acquire-valued-shoppers-challenge/" 
# Read train and test data and run Logistic regression

import numpy as np
import pylab as pl
import os
import csv
import random as rnd 
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import log_loss
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import scipy.sparse as sp
from pympler import tracker

loc_train = "//home//harini//Kaggle_data//Criteo//train.csv"
loc_test = "//home//harini//Kaggle_data//Criteo//test.csv"

feature_file = "//home//harini//Kaggle_data//Criteo//train_features_07242014.pkl"
pipeline_file = "//home//harini//Kaggle_data//Criteo//train_pipeline_07242014.pkl"

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

# Extract data into X and y
def getItems(fileName, itemsLimit=None):
    """ Reads data file. """
    
    with open(fileName) as items_fd:
        logging.info("Sampling...")
        if itemsLimit:
            numItems = 45000000
            sampleIndexes = set(rnd.sample(range(numItems),itemsLimit*10))
        
        #print(sampleIndexes,numItems)
        logging.info("Sampling done. Reading data...")
        itemReader=csv.DictReader(items_fd, delimiter=',')
        itemNum = 0
        poscount = 0
        negcount = 0
        if itemsLimit: 
            for i, item in enumerate(itemReader):
                if poscount < itemsLimit/2 and item["Label"]=="1":
                    itemNum += 1
                    poscount += 1
                    yield itemNum, item 
                if i in sampleIndexes and negcount < itemsLimit/2 and item["Label"]=="0":
                    itemNum += 1
                    negcount += 1
                    yield itemNum, item
                if itemNum >= itemsLimit:
                    break
        else:
            for i, item in enumerate(itemReader):
                    itemNum += 1
                    yield itemNum, item  

def processData(fileName, itemsLimit=None):  

    processMessage = ("Generate features for ")+os.path.basename(fileName)
    logging.info(processMessage+"...")

    m1=13
    m2=26
    Xcat_permissible = ["C5","C6","C8","C9","C14","C17","C20","C22","C23","C25"]
    replace_cat_clusters={}
    for cats in Xcat_permissible:
        replace_cat_clusters[cats]=0
        
    item_ids = []
    header = []

    colX=[]
    rowX=[]
    data_valsX= []

    colXcat=[]
    rowXcat=[]
    data_valsXcat= []
    replace_cat_indices = {}

    y=[]
    for processedCnt, item in getItems(fileName, itemsLimit):
        col=0
        col_cat=0
        for itemkey,itemval in item.iteritems():
            if processedCnt == 1 and not itemkey == "Id" and not itemkey == "Label":
                header.append(itemkey)
            if itemval == "":
                itemval = "NaN"
            if itemkey == "Id":
                item_ids.append(itemval)
            if itemkey == "Label":
                #y[processedCnt-1] = float(itemval)
                y.append(float(itemval))
            if itemkey in Xcat_permissible:
                #Xcat[processedCnt-1,col_cat] = (float(int(itemval, 16)) if not itemval == "NaN" else np.nan)
                if (itemkey,itemval) in replace_cat_indices:
                    rowXcat.append(processedCnt-1)
                    colXcat.append(col_cat)
                    data_valsXcat.append(replace_cat_indices[itemkey,itemval] if not itemval == "NaN" else np.nan)
                else:
                    replace_cat_clusters[itemkey] += 1
                    replace_cat_indices[itemkey,itemval] =  replace_cat_clusters[itemkey]
                    rowXcat.append(processedCnt-1)
                    colXcat.append(col_cat)
                    data_valsXcat.append(replace_cat_indices[itemkey,itemval] if not itemval == "NaN" else np.nan)
 
                col_cat +=1
            if "I" in itemkey and not itemkey == "Id":
                #X[processedCnt-1,col] = float(itemval)
                rowX.append(processedCnt-1)
                colX.append(col_cat)
                data_valsX.append((float(int(itemval, 16)) if not itemval == "NaN" else np.nan))
                col +=1
        if processedCnt%10000 == 0:                 
            logging.debug(processMessage+": "+str(processedCnt)+" items done")
            logging.debug("nnz elements: "+str(len(rowX)+len(rowXcat)))
    
    logging.info("Generating sparse array")   
    X = sp.csr_matrix((data_valsX,(rowX,colX)), shape=(processedCnt, m1), dtype=np.float)
    Xcat = sp.csr_matrix((data_valsXcat,(rowXcat,colXcat)), shape=(processedCnt, m2), dtype=np.float)
    
    logging.info("Imputing data for missing values")  
    imp = Imputer(missing_values="NaN", strategy='most_frequent', axis=0,verbose=10)
    imp.fit(X)
    X = imp.transform(X)
    imp2 = Imputer(missing_values="NaN", strategy='most_frequent', axis=0,verbose=10)
    imp2.fit(Xcat)
    Xcat = imp2.transform(Xcat)
    
    print replace_cat_clusters
    logging.info("Converting categorical data into dummy variables")    
    enc = OneHotEncoder(n_values='auto', categorical_features='all',sparse=True, dtype=np.float)
    Xcat = enc.fit_transform(Xcat.toarray())
    X = sp.hstack([X,Xcat])
    
    logging.debug("Size of train data is" + str(X.shape))
    logging.debug("Size of label data is" + str(len(y)))
    logging.debug("Number of positive examples is" + str(sum(y)))
    return (X,y,item_ids,header,imp, imp2,enc,replace_cat_clusters)

def processData_test(fileName, header, imp,imp2,enc, replace_cat_clusters, ml_algo,itemsLimit=100000):  

    processMessage = ("Generate features for ")+os.path.basename(fileName)
    logging.info(processMessage+"...")
    
    m1=13
    m2=26
    Xcat_permissible = ["C5","C6","C8","C9","C14","C17","C20","C22","C23","C25"]
       
    item_ids = []
    header = []

    colX=[]
    rowX=[]
    data_valsX= []

    colXcat=[]
    rowXcat=[]
    data_valsXcat= []
    replace_cat_indices = {}

    y=[]
    
    blockCount = 0
    predicted_scores=[]
    predicted_scores_curr = []
    item_ids = []
    #X=np.zeros([1,m1])
    #Xcat=np.zeros([1,m2])
    #tr = tracker.SummaryTracker()
    for processedCnt, item in getItems(fileName, itemsLimit=None):
        col=0
        col_cat=0
        #X=np.zeros([1,m1])
        #Xcat=np.zeros([1,m2])
        for itemkey,itemval in item.iteritems():
            if processedCnt == 1 and not itemkey == "Id" and not itemkey == "Label":
                header.append(itemkey)
            if itemval == "":
                itemval = "NaN"
            if itemkey == "Id":
                item_ids.append(itemval)
            if itemkey in Xcat_permissible:
                #Xcat[processedCnt-1,col_cat] = (float(int(itemval, 16)) if not itemval == "NaN" else np.nan)
                if (itemkey,itemval) in replace_cat_indices:
                    #Xcat[0,col_cat] = (float(int(itemval, 16)) if not itemval == "NaN" else np.nan)
                    rowXcat.append(blockCount)
                    colXcat.append(col_cat)
                    data_valsXcat.append(replace_cat_indices[itemkey,itemval] if not itemval == "NaN" else np.nan)
                else:
                    #Xcat[0,col_cat] = np.nan
                    rowXcat.append(blockCount)
                    colXcat.append(col_cat)
                    data_valsXcat.append(np.nan)
                col_cat +=1
            if "I" in itemkey and not itemkey == "Id":
                #X[0,col] = float(itemval)
                rowX.append(blockCount)
                colX.append(col_cat)
                data_valsX.append((float(int(itemval, 16)) if not itemval == "NaN" else np.nan))
                col +=1
                
            
        blockCount += 1
        if blockCount>=itemsLimit:
            logging.debug(processMessage+": "+str(processedCnt)+" items done")
            logging.debug("nnz elements: "+str(len(rowX)+len(rowXcat)))
    
            logging.info("Generating sparse array")   
            X = sp.csr_matrix((data_valsX,(rowX,colX)), shape=(blockCount, m1), dtype=np.float64)
            Xcat = sp.csr_matrix((data_valsXcat,(rowXcat,colXcat)), shape=(blockCount, m2), dtype=np.float64)
            
            blockCount=0        
            colX=[]
            rowX=[]
            data_valsX= []

            colXcat=[]
            rowXcat=[]
            data_valsXcat= []    
            logging.info("Imputing data for missing values")  
            X = imp.transform(X)
            Xcat = imp2.transform(Xcat) 
            
            logging.info("Converting categorical data into dummy variables")   
            Xcat = enc.transform(Xcat.toarray())
            X = sp.hstack([X,Xcat])
        #        X = np.concatenate((X,Xcat.toarray()),axis=1)

            logging.info("Predicting test data")    
            X = StandardScaler(with_mean=False).fit_transform(X)
            #temp = ml_algo.decision_function(X) - np.log(3)
            #predicted_scores_curr = 1/(1+np.exp(-1*temp))
            predicted_scores_curr = ml_algo.predict_proba(X).T[1]
            predicted_scores = np.append(predicted_scores,predicted_scores_curr)  
            del X, Xcat, predicted_scores_curr
            #tr.print_diff()     

    X = sp.csr_matrix((data_valsX,(rowX,colX)), shape=(blockCount, m1), dtype=np.float64)
    Xcat = sp.csr_matrix((data_valsXcat,(rowXcat,colXcat)), shape=(blockCount, m2), dtype=np.float64)
 
    X = imp.transform(X)
    Xcat = imp2.transform(Xcat)
    
    logging.info("Converting categorical data into dummy variables")    
    Xcat = enc.transform(Xcat.toarray())
    X = sp.hstack([X,Xcat])
    #X = np.concatenate((X,Xcat), axis=1)

    X = StandardScaler(with_mean=False).fit_transform(X)
    predicted_scores_curr = ml_algo.predict_proba(X).T[1]
    predicted_scores = np.append(predicted_scores,predicted_scores_curr)        
    return predicted_scores, item_ids        
    
  
#logging.info("Extracting training data")
#X, y, ids, header, imp, imp2, enc, replace_cat_clusters = processData(loc_train, itemsLimit=500000)
#[N,m] = X.shape

#logging.info("Saving training data")
#joblib.dump((X, y, ids, header), feature_file) 
#joblib.dump((imp, imp2, enc, replace_cat_clusters), pipeline_file)

logging.info("Loading training data")
X, y, ids, header = joblib.load(feature_file)
imp, imp2, enc, replace_cat_clusters = joblib.load(pipeline_file)
[N,m] = X.shape
logging.info("normalizing training data")
X = StandardScaler(with_mean=False).fit_transform(X)
y = np.ravel(y)

logging.info("Feature preparation done, fitting model...")
loss="log"
cv=ShuffleSplit(N, n_iter=10, test_size=0.5, train_size=None, random_state=None)
tuned_parameters = [{'C': [10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)]}]
#tuned_parameters = [{'C': [10**(-4)]}]
class_weight={}
class_weight[0]=4
class_weight[1]=1
clf_LR = LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=class_weight, random_state=None)
clf = GridSearchCV(clf_LR, tuned_parameters, scoring='log_loss', cv=cv,refit=True,verbose=True)
clf.fit(X, y)

logging.info("Printing cross-validation log-loss scores")
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() / 2, params))        

logging.info("Fitting best estimator")
clf.best_estimator_.fit(X,y)
 
logging.info("Reading coefficients of best estimator")
coef_LR = clf.best_estimator_.coef_.ravel()

logging.info("Printing coefficients of best estimator")
count=0
out_coeffs = open("coeffs_07242014.csv", "w+")
for e,val in enumerate(coef_LR):
    out_coeffs.write("%d,%1.10f\n" % (e,val))
out_coeffs.close()    
    
logging.info("Clearing train data")    
X = []
y = []

logging.info("predicting test data")
predicted_scores, item_ids_test = processData_test(loc_test,header,imp,imp2,enc,replace_cat_clusters, clf)

logging.info("writing submission file")
submission_file="criteo_submission_07242014.csv"
outfile = open(submission_file, "w+")
outfile.write( "Id,Predicted\n" )
for e,val in enumerate(item_ids_test):
    outfile.write("%s,%1.10f\n" % (val,predicted_scores[e]))
outfile.close()


