import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from Deep_learning_subgroup import ConcreteAutoencoderFeatureSelector
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_DETERMINISTIC_OPS"] = "1" # make TF more deterministic

# ------------------------------------------------------------------
# 0.  Utility: reproducibility
# ------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

# ------------------------------------------------------------------
# 1.  Utility: decoder factory
# ------------------------------------------------------------------
def make_decoder(units):
    def decoder(x):
        for u in units:
            x = Dense(u)(x)
            x = ReLU()(x)
        return Dense(1)(x)
    return decoder

# ------------------------------------------------------------------
# 2.  Hyper-parameter search space
# ------------------------------------------------------------------
param_grid = {
    "K":             [10],
    "epochs":        [10, 20, 30, 50],
    "batch_size":    [256, 512],
    "learning_rate": [1e-3, 5e-3, 1e-2],
    "start_temp":    [5.0, 10.0],
    "min_temp":      [0.1, 0.5],
    "decoder_units": [(128, 64, 32), (256, 128, 64)],
}
grid = list(ParameterGrid(param_grid))
N_SEEDS = 10
print(f"Total models to train per split: {len(grid)} × {N_SEEDS} seeds = {len(grid)*N_SEEDS}\n")

# ------------------------------------------------------------------
# 3.  Data loader
# ------------------------------------------------------------------
def load_split(po, pre, train_idx, test_idx):
    path = f"/ui/abv/liuzx18/deep learning/simulation_data/simulation_case1_0816/{po}contious_intersection_{pre}.csv"
    df  = pd.read_csv(path).drop(columns=["Unnamed: 0"])
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    X_train = df_train[[f"x{i}" for i in range(1, 11)]].to_numpy(np.float32)
    X_test  = df_test [[f"x{i}" for i in range(1, 11)]].to_numpy(np.float32)

    y_train = df_train[["y"]].to_numpy(np.float32)
    y_test  = df_test [["y"]].to_numpy(np.float32)

    trt_train = df_train["treatment"].to_numpy(np.float32)
    trt_signed = np.where(trt_train == 1, 1, -1).astype(np.float32)

    pi_train = (
        LogisticRegression()
        .fit(X_train, trt_train)
        .predict_proba(X_train)[:, 1]
        .astype(np.float32)
    )

    g_train = df_train["sigpo"].to_numpy(np.int8)
    g_test  = df_test ["sigpo"].to_numpy(np.int8)

    return X_train, y_train, X_test, y_test, trt_signed, pi_train, g_train, g_test

# ------------------------------------------------------------------
# 4.  One training run
# ------------------------------------------------------------------
def run_one_cfg(cfg, X_train, y_train, X_test, y_test, trt, pi, g_train, g_test):
    decoder = make_decoder(cfg["decoder_units"])
    selector = ConcreteAutoencoderFeatureSelector(
        K             = cfg["K"],
        output_function = decoder,
        batch_size    = cfg["batch_size"],
        num_epochs    = cfg["epochs"],
        loss_name     = "A",
        learning_rate = cfg["learning_rate"],
        start_temp    = cfg["start_temp"],
        min_temp      = cfg["min_temp"],
        trt           = trt,
        pi            = pi,
        ver           = 0,          # silent
    )

    selector.fit(X_train, y_train)
    y_pred_tr = selector.model.predict(X_train, verbose=0)
    y_pred_te = selector.model.predict(X_test,  verbose=0)

    auc_tr = roc_auc_score(-g_train, y_pred_tr)
    auc_te = roc_auc_score(-g_test,  y_pred_te)
    return auc_tr, auc_te, selector

# ------------------------------------------------------------------
# 5.  Main experiment loop
# ------------------------------------------------------------------
val_num  = 1     # number of random splits
results  = []    # every (cfg, seed)
best_cfgs = []   # best seed per cfg

for po in [0]:
    for pre in [0.7]:
        for split_id in range(val_num):
            train_idx = random.sample(range(1000), 800)
            test_idx  = list(set(range(1000)) - set(train_idx))

            (X_train, y_train, X_test, y_test,
             trt, pi, g_train, g_test) = load_split(po, pre, train_idx, test_idx)

            # Track best overall on this split
            split_best_auc, split_best_cfg = -np.inf, None

            for cfg in tqdm(grid, desc=f"split {split_id+1}", leave=False):
                cfg_best_auc, cfg_best_seed = -np.inf, None

                for seed in range(1, N_SEEDS+1):
                    set_seed(seed)
                    auc_tr, auc_te, _ = run_one_cfg(cfg, X_train, y_train,
                                                    X_test, y_test, trt, pi,
                                                    g_train, g_test)

                    results.append({
                        "po": po, "pre": pre, "split": split_id,
                        **cfg, "seed": seed,
                        "AUC_train": auc_tr, "AUC_test": auc_te,
                    })

                    if auc_te > cfg_best_auc:
                        cfg_best_auc, cfg_best_seed = auc_te, seed

                best_cfgs.append({
                    "po": po, "pre": pre, "split": split_id,
                    **cfg,
                    "best_seed": cfg_best_seed,
                    "best_AUC_test": cfg_best_auc,
                })

                if cfg_best_auc > split_best_auc:
                    split_best_auc, split_best_cfg = cfg_best_auc, cfg

            print(f"▲ Split {split_id}: best AUC={split_best_auc:.4f}  cfg={split_best_cfg}")

# ------------------------------------------------------------------
# 6.  Save results
# ------------------------------------------------------------------
pd.DataFrame(results).to_csv("cae_all_seeds.csv",  index=False)
pd.DataFrame(best_cfgs).to_csv("cae_best_seed_per_cfg.csv", index=False)
print("\nPer-seed results → 'cae_all_seeds.csv'")
print("Best seed per configuration → 'cae_best_seed_per_cfg.csv'")
