import lightgbm as lgb
import matplotlib.pyplot as plt
import random
import umap.plot
from collections import Counter
import pandas as pd 
import numpy as np 
import warnings
import argparse
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import statsmodels.api as sm
from statsmodels.formula.api import ols
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re

def squared_log_continues(predt,dtrain):

    #print(predt.shape)
    grad = gradient_continues(predt, dtrain)
    hess = hessian_continues(predt, dtrain)
    return grad, hess
def gradient_continues(predt, dtrain):
    y = dtrain.get_label()
    
    c=(X_train_trt+1.0)/2.0-pi_train
    #c=1
    return -2.0*c*(y-predt*c)

def hessian_continues(predt, dtrain):
    
    y = dtrain.get_label()
    
    c=(X_train_trt+1.0)/2.0-pi_train
    
    return 2.0*(c**2)


def rmsle_continues(predt, dtrain):
 
    y = dtrain.get_label()
    y_real=y.reshape((y.shape[0],))
    
    
    c=(X_train_trt+1.0)/2.0-pi_train
    
    elements = (y-c*predt)**2
    
    acc=(predt-x_train_trt_effect)**2
    #print (float(np.sqrt(np.sum(acc) / len(y))))
    
    return 'LOSS', float(np.sqrt(np.sum(elements) / len(y)))

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    
df_result_box=pd.DataFrame()
val_num=1

for po in [0,1,2]:
    for pre in [0.1,0.5,0.7,1,2,3]:
        train_list=[]
        test_list=[]

        for kk in range (val_num):
            print(kk)
            train_num=random.sample(range(0,1000), 800)
            test_num=list(set(range(0,1000))-set(train_num))

            data_simulation= pd.read_csv('/data_path/{}contious_intersection_{}.csv'.format(po,pre)).iloc[train_num,]
            data_simulation_test= pd.read_csv('/data_path/{}contious_intersection_{}.csv'.format(po,pre)).iloc[test_num,]
            del data_simulation['Unnamed: 0']
            del data_simulation_test['Unnamed: 0']
            x_df=data_simulation[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
            X_=np.array(x_df)[:]


            x_df_test=data_simulation_test[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
            X_test=np.array(x_df_test)

            y_train=np.array(data_simulation[['y']])
            y_train=y_train.reshape(800,)


            y_test=np.array(data_simulation_test[['y']])
            y_test=y_test.reshape(200,)
            print(y_test.shape)
            trt_=data_simulation[['treatment']]


            g_real=data_simulation[['sigpo']]
            g_real_test=data_simulation_test[['sigpo']]




            logreg = LogisticRegression()
            logreg.fit(X_,trt_)
            pi_x = logreg.predict_proba(X_)
            pi_train=pi_x[:,1]



            X_train_trt=np.where(trt_==1,1,-1)
            X_train_trt=X_train_trt.reshape(800,)

            dtrain = xgb.DMatrix(X_, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)


            results = {}
            par={'learning_rate':0.005,
             'verbosity':1,
             'booster':'gbtree',
             'max_depth':1,
             'lambda':5,
             'tree_method':'hist' 
            }
            model=xgb.train(par,
                  dtrain=dtrain,
                  num_boost_round=1000,
                  obj=squared_log_continues,
                  feval=rmsle_continues,
                  #evals=[(dtrain, 'dtrain')],
                  )


            pred_train = model.predict(dtrain)
            pred_00=1.0/(1.0+np.exp(-pred_train))
            pred_label=np.where(pred_00>0.5,1,0)
            auc_train = metrics.roc_auc_score(g_real.astype(int), pred_00)
            
            train_list.append(auc_train)


            pred_test = model.predict(dtest)
            pred_01=1.0/(1.0+np.exp(-pred_test))
            pred_label=np.where(pred_01>0.5,1,0)
            auc_test = metrics.roc_auc_score(g_real_test.astype(int), pred_01)
            
            test_list.append(auc_test)

            f_importance = model.get_score(importance_type='gain')
            f_importance={k: v for k, v in sorted(f_importance.items(), key=lambda item: item[1])}
            importance_df = pd.DataFrame.from_dict(data=f_importance, 
                                               orient='index')
            if kk==0:
                rank_feature_top2=list(importance_df[0].index)[-2:]
            else:
                rank_feature_top2=rank_feature_top2+list(importance_df[0].index)[-2:]
            
            
        dic_r={}
        for d in np.unique(rank_feature_top2):
            dic_r[d]=Counter(rank_feature_top2)[d]/float(val_num)
        dic_r


        dic=pd.DataFrame(dic_r.items())
        dic.columns=['feature','precentage']
        df_result_box['com_{}_{}'.format(po,pre)]=test_list




    


  
