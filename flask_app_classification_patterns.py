import pandas as pd
import datetime as dt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from flask import Flask ,render_template
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import cross_validate
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import plot_precision_recall_curve

import numpy as np
from time import time


app = Flask(__name__)
matplotlib.use('Agg')

FILE_NAME="/home/ubuntu/Wangiri_Detection/wangiri_data_20210712.tsv"
chosen_pattern="p3"
cross_val_splits=10
balanceData= True # Change to False to use normal data
sampling_ratio= 0.5

# Pattern 1
def get_missedcall_pattern_groundtruth(data):
    data['Label'] = np.where((data['c_number'] != data['a_number'])  &  (data["conversation_time"]<30), 1, 0)
    return data
# Filter suspicious numbers for p2 and p3
def get_suspicious_nos(data):
    bnumcount= data.groupby(['a_number'])['b_number'].nunique()
    callcount=data.groupby(['a_number']).count()["b_number"]
    plot_val = {'anum':callcount.index, 'callcount': callcount.values , 'bnumcount':bnumcount.values}  
    count_data=pd.DataFrame(plot_val)
    suspicious_nums_out=count_data[(count_data["bnumcount"]>=200) | (count_data["callcount"]>=200) ]
    return suspicious_nums_out

# Pattern 2    
def get_callback_pattern_groundtruth(data):
    suspicious_nums_out=get_suspicious_nos(data)
    targettednums=data[data["a_number"].isin(suspicious_nums_out["anum"])]["b_number"].unique()
    callback=data[(data["a_number"].isin(targettednums)) & (data["b_number"].isin(suspicious_nums_out["anum"].values))]
    data['Label'] = np.where((data["a_number"].isin(callback["a_number"].unique())), 1, 0)
    return data
# Pattern 3
def get_broadcast_seq_groundtruth(data):
    suspicious_nums_out=get_suspicious_nos(data)
    data['Label'] = np.where((data["a_number"].isin(suspicious_nums_out["anum"])), 1, 0)
    return data




def read_data(file):
    print ("reading begin ")
    data = pd.read_table(file)
    print ("reading end ")
    return data

def preprocess_data(data):
    data.columns = data.columns.str.strip()
    print ("preprocessing begin ")
    #data.drop(columns=["calling_number","dialled_number"], inplace=True)
    data.drop(columns=["country_code_a","country_code_b"], inplace=True)
    data.dropna(inplace=True)
    data['start_time']=pd.to_datetime(data['start_time']).map(dt.datetime.toordinal)
    print ("preprocessing begin ")
    return data

def trtest_split(data):
    # mixed data preparation for iforest
    X = data.drop('Label', 1)
    y = data['Label']
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(X)
    X_train, X_test,y_train, y_test = train_test_split(scaled_data,y, test_size=0.2, random_state=22)
    print(X.shape,y.value_counts())
    

    return  X,y,X_train, X_test,y_train, y_test
    
    
def balanced_data(data):
    print ("Balancing begin")
    X = data.drop('Label', 1)
    y = data['Label']
    oversample = RandomOverSampler(sampling_strategy=sampling_ratio)
    X_over, y_over = oversample.fit_resample(X, y)
    print ("Balancing complete")
   
    #for classification
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(X_over)
    X_train, X_test,y_train, y_test = train_test_split(scaled_data,y_over, test_size=0.2, random_state=22)

    print(X_over.shape,y_over.value_counts())
   
    
    return scaled_data,y_over,X_train, X_test,y_train, y_test


def confusion_matrix_scorer_cv(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
             'fn': cm[1, 0], 'tp': cm[1, 1]} 

def confusion_matrix_classification(cv_results, name):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    cf_cv_matrix=np.array([[np.sum(cv_results['test_tn']),np.sum(cv_results['test_fp'])],[np.sum(cv_results['test_fn']),np.sum(cv_results['test_tp'])]])
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_cv_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_cv_matrix.flatten()/np.sum(cf_cv_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    
    labels = np.asarray(labels).reshape(2,2)
    plt.clf()
    sns.heatmap(cf_cv_matrix, annot=labels , fmt='', annot_kws={"size": 16})
    plt.title(name)
    plt.savefig('static/images/'+chosen_pattern+'/'+name+'.png')
    return cf_cv_matrix

def conf_matrix(ytest ,predicted , name):
    cf_matrix= confusion_matrix(ytest, predicted)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    plt.clf()
    sns.heatmap(cf_matrix, annot=labels , fmt='', annot_kws={"size": 16})
    plt.title(name)
    plt.savefig('static/images/'+chosen_pattern+'/'+name+'.png')
    
def calc_eval_metrics(cf_test_matrix):
    tp= cf_test_matrix[1][1]
    tn= cf_test_matrix[0][0]
    fp = cf_test_matrix[0][1]
    fn = cf_test_matrix[1][0]
    total = tp + fp +fn+ tn
    p= tp + fn
    n= tn +fp
    acc = (tp + tn)/total
    prec = tp / ( tp + fp ) 
    rec = tp / p
    f1 = 2 * prec* rec/ (prec+rec)
    return acc,prec,rec,f1
# For getting precision recall curve
def get_pr_recall():
    #print("in pr curve model fitting")
    #modelNB = GaussianNB().fit(X_train,y_train)
    #modelRC = RandomForestClassifier(n_estimators=150).fit(X_train,y_train)
    #modelxgb = xgb.XGBClassifier().fit(X_train,y_train)
    print("in pr curve model fitting end")


    #plot_precision_recall_curve(modelNB,X_test,y_test,ax=plt.gca(),name="Naive Bayes")
    #plot_precision_recall_curve(modelRC,X_test,y_test,ax=plt.gca(),name="Random Forest")
    #plot_precision_recall_curve(modelRC,X_test,y_test,ax=plt.gca(),name="XGBoost")
    #plt.savefig('Precison recall curves')
    #plt.savefig('static/images/PRcurves/'+chosen_pattern+'PRbal.png')
    print("curve finished")


@app.route('/')
@app.route('/home', methods=["GET", "POST"])
def home():
    return render_template('home.html')

@app.route('/naiveBayes', methods=["GET", "POST"])
def naiveBayes():
    try:
        t0=time()
        modelNB = GaussianNB().fit(X_train,y_train)
        
        print ("Naive Bayes model ")
        
        y_pred_train = modelNB.predict(X_train)
 
        y_pred = modelNB.predict(X_test)
        t1=time()
        print("Time of execution NB",(t1-t0))
        conf_matrix(y_train,y_pred_train , "naiveBayes_performance_train_nocv_nobal")
 
        conf_matrix(y_test, y_pred , "naiveBayes_performance_test_nocv_nobal")
 
        accuracy, f1 , precision , recall = "{0:.2%}".format(accuracy_score(y_test,y_pred)), "{0:.2%}".format(f1_score(y_test,y_pred)), "{0:.2%}".format(precision_score(y_test,y_pred)) , "{0:.2%}".format(recall_score(y_test,y_pred))
 
        #cv_results = cross_validate(modelNB,X, y, cv=cross_val_splits
        #                     ,scoring=confusion_matrix_scorer_cv)
        #cf_test_matrix = confusion_matrix_classification(cv_results , "naiveBayes_performance_test")
        #t1=time()
        #print("Time of execution Naive Bayes",(t1-t0))
        #acc,prec,rec,f1=calc_eval_metrics(cf_test_matrix)
        #accuracy, f1 , precision , recall = "{0:.2%}".format(acc), "{0:.2%}".format(f1), "{0:.2%}".format(prec) , "{0:.2%}".format(rec)
        return render_template('home.html', url_test ='/static/images/'+chosen_pattern+'/naiveBayes_performance_test_nocv_nobal.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Naive Bayes" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
    
@app.route('/randomForest', methods=["GET", "POST"])
def randomForest():
    try:
        print ("Random Forest model fit")
        t0=time()

        modelRC = RandomForestClassifier(n_estimators=150).fit(X_train,y_train)
        plot_precision_recall_curve(modelRC,X_test,y_test)
        plt.savefig('static/images/PRcurves/'+chosen_pattern+'/rfpr.png')
 
        y_pred_train = modelRC.predict(X_train)
 
        y_pred = modelRC.predict(X_test)
        t1=time()
        print("Time of execution Random Forest",(t1-t0))
        conf_matrix(y_train,y_pred_train , "randomForest_performance_train")
 
        conf_matrix(y_test, y_pred , "randomForest_performance_test")
 
        accuracy, f1 , precision , recall = "{0:.2%}".format(accuracy_score(y_test,y_pred)), "{0:.2%}".format(f1_score(y_test,y_pred)), "{0:.2%}".format(precision_score(y_test,y_pred)) , "{0:.2%}".format(recall_score(y_test,y_pred))
 
        print("Random Forest results", accuracy, f1 , precision , recall)
 
        return render_template('home.html', url_test ='/static/images/'+chosen_pattern+'/randomForest_performance_test.png' ,url_train= '/static/images/'+chosen_pattern+'/randomForest_performance_train.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Random Forest" )
 
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
@app.route('/SVC', methods=["GET", "POST"])
def SVC():
    try:
        print ("SVC model fit")
        modelSVC = svm.SVC(C=1.5)
        cv_results = cross_validate(modelSVC,X, y, cv=cross_val_splits
                                    ,scoring=confusion_matrix_scorer_cv)
        cf_test_matrix = confusion_matrix_classification(cv_results , "SVM_performance_test")
        acc,prec,rec,f1=calc_eval_metrics(cf_test_matrix)
        accuracy, f1 , precision , recall = "{0:.2%}".format(acc), "{0:.2%}".format(f1), "{0:.2%}".format(prec) , "{0:.2%}".format(rec)
        print("SVC results", accuracy, f1 , precision , recall)
        return render_template('home.html', url_test ='/static/images/'+chosen_pattern+'/SVM_performance_test.png', accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Support Vector Machine" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
@app.route('/xgboost', methods=["GET", "POST"])
def xgboost():
    try:
        print ("XGBoost model fit")
        t0=time()
        modelxgb = xgb.XGBClassifier().fit(X_train,y_train)
        plot_precision_recall_curve(modelxgb,X_test,y_test)
        plt.savefig('static/images/PRcurves/'+chosen_pattern+'/xgbpr.png')
        #cv_results = cross_validate(modelxgb,X_train,y_train, cv=cross_val_splits
        #                            ,scoring=confusion_matrix_scorer_cv)
        #cf_test_matrix = confusion_matrix_classification(cv_results , "xgboost_performance_test")
        #acc,prec,rec,f1=calc_eval_metrics(cf_test_matrix)
        #accuracy, f1 , precision , recall = "{0:.2%}".format(acc), "{0:.2%}".format(f1), "{0:.2%}".format(prec) , "{0:.2%}".format(rec)
        y_pred_train = modelxgb.predict(X_train)
 
        y_pred = modelxgb.predict(X_test)
        t1=time()
        print("Time of execution XGBoost",(t1-t0))
        conf_matrix(y_train,y_pred_train , "xgboost_performance_train")
 
        conf_matrix(y_test, y_pred , "xgboost_performance_test")
 
        accuracy, f1 , precision , recall = "{0:.2%}".format(accuracy_score(y_test,y_pred)), "{0:.2%}".format(f1_score(y_test,y_pred)), "{0:.2%}".format(precision_score(y_test,y_pred)) , "{0:.2%}".format(recall_score(y_test,y_pred))
 
 
        print("xgboost results", accuracy, f1 , precision , recall)
        return render_template('home.html', url_test ='/static/images/'+chosen_pattern+'/xgboost_performance_test.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="XGBoost Algorithm" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
    
data=read_data(FILE_NAME)
datapreproc=preprocess_data(data)
print("Chosen pattern:"+chosen_pattern)
if chosen_pattern=="p1":
    labelled_data=get_missedcall_pattern_groundtruth(datapreproc)
elif chosen_pattern=="p2":
    labelled_data=get_callback_pattern_groundtruth(datapreproc)
else:
    labelled_data=get_broadcast_seq_groundtruth(datapreproc)

    
if balanceData:
    X,y,X_train, X_test,y_train, y_test = balanced_data(labelled_data)
else:
    X,y,X_train, X_test,y_train, y_test = trtest_split(labelled_data)
    
#get_pr_recall()
