import numpy as np
import lightgbm as lgb


def params(cfg, dataset, n):
    sec = cfg["lightgbm"]
    lr, trees = sec.getfloat("learning_rate"), sec.getint("n_estimators")
    if dataset == "kmu_tls":
        lr, trees = sec.getfloat("kmu_tls_learning_rate"), sec.getint("kmu_tls_n_estimators")
    elif dataset.startswith("cesnet") and n <= sec.getint("cesnet_small_n"):
        lr = sec.getfloat("cesnet_small_n_learning_rate")
    return lr, trees


def fit(cfg, dataset, n, seed, F, y, nc):
    sec = cfg["lightgbm"]
    lr, trees = params(cfg, dataset, n)
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=trees, learning_rate=lr, num_leaves=sec.getint("num_leaves"),
                           min_child_samples=sec.getint("min_child_samples"), reg_lambda=sec.getfloat("reg_lambda"),
                           max_bin=sec.getint("max_bin"), random_state=seed, n_jobs=cfg.getint("experiment", "n_jobs"), verbose=-1)
    m.fit(F, y)
    return m


def predict(m, F):
    return m.predict(F)
