import os
os.environ['PYTHONHASHSEED']=str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import warnings
import argparse
import pandas as pd
import numpy as np
from Deep_learning_subgroup import ConcreteAutoencoderFeatureSelector
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU,ReLU
from keras import backend as K
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
warnings.filterwarnings("ignore")
def ini_set(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
val_num=1
df_result_box=pd.DataFrame()
for po in [0]:
    for pre in [0.1]:
        best_train=[]
        best_test=[]
        best_feature=[]
        best_seed=[]
        for kk in range (val_num):
            ini_set(kk+100)
            train_num=random.sample(range(0,1000), 800)
            test_num=list(set(range(0,1000))-set(train_num))
            max_auc=0
            
            ini_set(21)
            data_simulation= pd.read_csv('/data_path/{}contious_intersection_{}.csv'.format(po,pre)).iloc[train_num,]
            data_simulation_test= pd.read_csv('/data_path/{}contious_intersection_{}.csv'.format(po,pre)).iloc[test_num,]
            del data_simulation['Unnamed: 0']
            del data_simulation_test['Unnamed: 0']
            x_df=data_simulation[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
            X_=np.array(x_df).astype(np.float)
            x_df_test=data_simulation_test[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
            X_test=np.array(x_df_test).astype(np.float)
            y_dummy=np.array(data_simulation[['y']]).astype(np.float)

            y_dummy=y_dummy.reshape(data_simulation.shape[0],1)

            y=y_dummy

            trt_=data_simulation[['treatment']]
            g_real=data_simulation[['sigpo']]
            g_real_test=data_simulation_test[['sigpo']]

            logreg = LogisticRegression()
            logreg.fit(X_,trt_)
            pi_x = logreg.predict_proba(X_)
            pi_train=pi_x[:,1]

            X_train_trt=np.where(trt_==1,1,-1)

            feature_select={}
            #loss='binary_crossentropy'
            #loss='weight'
            loss='A'
            ep=10
            lr=0.1
            def decoder(x):
                x = Dense(128)(x)
                x = ReLU()(x)
                #x = Dropout(0.1)(x)
                x = Dense(64)(x)
                x = ReLU()(x)
                x = Dense(32)(x)
                x = ReLU()(x)
                #x = Dropout(0.1)(x)
                x = Dense(1)(x)

                return x

            result=[]
            selector = ConcreteAutoencoderFeatureSelector(K = 8, output_function = decoder,batch_size = 1000, num_epochs =ep,loss_name=loss,learning_rate = lr, start_temp = 10.0, min_temp = 0.5,trt=X_train_trt.astype(np.float),pi=pi_train.astype(np.float),ver=0)
            model_1=selector.fit(X_, y, X_, y)
            #selector.fit(geno_np_train, y, geno_np_train,y)
            #selector.fit(K.eval(geno_np_train), K.eval(y_noise_train), K.eval(geno_np_train), K.eval(y_noise_train))
            y_pred=model_1.model.predict(X_)
            auc = roc_auc_score(g_real.astype(int), y_pred)
            #print('auc {}',format(auc))
            y_pred_test=model_1.model.predict(X_test)
            auc_test = roc_auc_score(g_real_test.astype(int), y_pred_test)
            #print('test auc {}',format(auc_test))

           
        best_f=selector.get_support(indices = True)
      
        best_feature.append(best_f)

       


        df_result_box['com_{}_{}'.format(po,pre)]=best_test
        
        rank_feature=[]
        for j in best_feature:
            dic_result={}
            for k in np.unique(j):
                dic_result[str(k)]=Counter(j)[k]
            rank_feature.append(sorted(dic_result, key=dic_result.get, reverse=True))


        dic_r={}
        result_top2=rank_feature[0][0:2]
        for d in range(1,len(rank_feature)):
            result_top2=result_top2+rank_feature[d][0:2]
        rank_feature_top2=[int(i) for i in result_top2]
        for d in np.unique(rank_feature_top2):
            dic_r[str(d)]=Counter(rank_feature_top2)[d]/float(len(rank_feature))
        print('feature importance')
        print(dic_r)
        
        print('training auc')
        print(auc)
        
        print('testing auc')
        print(auc_test)

        dic=pd.DataFrame(dic_r.items())
        dic.columns=['feature','precentage']
        
        
        

        
        
