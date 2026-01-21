import os
os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import warnings
import pandas as pd
import numpy as np
from Deep_learning_subgroup import ConcreteAutoencoderFeatureSelector
from collections import Counter
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, ReLU
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def ini_set(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(),
        config=session_conf
    )
    K.set_session(sess)


# Number of simulation replications.
# Ideally, one might generate multiple independent simulated datasets.
# For simplicity, we use a single simulated dataset and repeat the experiment
# by varying the random seed used for train–test splitting.
val_num = 10


df_result_box = pd.DataFrame()

# ----------------------------
# Simple hyperparameter grid
#   - 2–3 configurations of hidden-layer sizes
#   - 2 learning rates
# Ideally, additional hyperparameters could be tuned as in the original paper,
# but we restrict the search space here for computational simplicity.
# for more extensive hyperparameter
# tuning, we strongly recommend running the code on high-performance
# computing (HPC) resources.
# ----------------------------
HP_GRID = [
    {"h1": 128, "h2": 64, "h3": 32, "lr": 0.1},
    {"h1": 64,  "h2": 32, "h3": 16, "lr": 0.1},
    {"h1": 128, "h2": 64, "h3": 32, "lr": 0.01},
    {"h1": 64,  "h2": 32, "h3": 16, "lr": 0.01},
]

for po in [1]:
    for pre in [0.1]:
        best_train = []
        best_test = []
        best_feature = []
        best_seed = []
        best_hp = [] 

        for kk in range(val_num):

            ini_set(kk + 100)
            train_num = random.sample(range(0, 1000), 800)
            test_num = list(set(range(0, 1000)) - set(train_num))
            data_path = 'datapath/{}contious_intersection_{}.csv'.format(po, pre)
            data_all = pd.read_csv(data_path)

            data_simulation = data_all.iloc[train_num, :].copy()
            data_simulation_test = data_all.iloc[test_num, :].copy()

            if 'Unnamed: 0' in data_simulation.columns:
                del data_simulation['Unnamed: 0']
            if 'Unnamed: 0' in data_simulation_test.columns:
                del data_simulation_test['Unnamed: 0']

            x_cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
            X_ = np.array(data_simulation[x_cols]).astype(np.float)
            X_test = np.array(data_simulation_test[x_cols]).astype(np.float)

            y_dummy = np.array(data_simulation[['y']]).astype(np.float).reshape(data_simulation.shape[0], 1)
            y = y_dummy

            trt_ = data_simulation[['treatment']]
            g_real = data_simulation[['sigpo']]
            g_real_test = data_simulation_test[['sigpo']]

            # propensity model
            logreg = LogisticRegression()
            logreg.fit(X_, trt_)
            pi_x = logreg.predict_proba(X_)
            pi_train = pi_x[:, 1]

            X_train_trt = np.where(trt_ == 1, 1, -1)

            loss = 'A'
            ep = 10
            max_auc = -1
            best_auc = None
            best_auc_test = None
            best_f = None
            best_i = None
            best_hp_cfg = None
            # Loop over different random initializations to account for
            # The final model is selected based on training performance only 
            for i in range(1, 30):
                ini_set(i)
                for hp in HP_GRID:
                    h1, h2, h3, lr = hp["h1"], hp["h2"], hp["h3"], hp["lr"]

                    def decoder(x, h1=h1, h2=h2, h3=h3):
                        x = Dense(h1)(x)
                        x = ReLU()(x)
                        x = Dense(h2)(x)
                        x = ReLU()(x)
                        x = Dense(h3)(x)
                        x = ReLU()(x)
                        x = Dense(1)(x)
                        return x

                    selector = ConcreteAutoencoderFeatureSelector(
                        K=8,
                        output_function=decoder,
                        batch_size=1000,
                        num_epochs=ep,
                        loss_name=loss,
                        learning_rate=lr,
                        start_temp=10.0,
                        min_temp=0.5,
                        trt=X_train_trt.astype(np.float),
                        pi=pi_train.astype(np.float),
                        ver=0
                    )

                    model_1 = selector.fit(X_, y, X_, y)

                    y_pred = model_1.model.predict(X_)
                    auc = roc_auc_score(g_real.astype(int), y_pred)

                    y_pred_test = model_1.model.predict(X_test)
                    auc_test = roc_auc_score(g_real_test.astype(int), y_pred_test)

                    if auc >= max_auc:
                        max_auc = auc
                        best_i = i
                        best_hp_cfg = hp
                        best_auc = auc
                        best_auc_test = auc_test
                        best_f = selector.get_support(indices=True)

            best_train.append(best_auc)
            best_test.append(best_auc_test)
            best_feature.append(best_f)
            best_seed.append(best_i)
            best_hp.append(best_hp_cfg)

        df_result_box['com_{}_{}'.format(po, pre)] = best_test
        rank_feature = []
        for j in best_feature:
            dic_result = {}
            for k in np.unique(j):
                dic_result[str(k)] = Counter(j)[k]
            rank_feature.append(sorted(dic_result, key=dic_result.get, reverse=True))

        dic_r = {}
        result_top2 = rank_feature[0][0:2]
        for d in range(1, len(rank_feature)):
            result_top2 = result_top2 + rank_feature[d][0:2]

        rank_feature_top2 = [int(i) for i in result_top2]
        for d in np.unique(rank_feature_top2):
            dic_r[str(d)] = Counter(rank_feature_top2)[d] / float(len(rank_feature))

       

        dic = pd.DataFrame(dic_r.items())
        dic.columns = ['feature', 'precentage']

        
        print("chosen hyperparameters frequency (by train-selected model):")
        print(best_hp)

print('Testing AUC importance')
print(df_result_box)
print('feature importance')
print(dic)