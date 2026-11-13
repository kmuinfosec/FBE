import configparser
import os


def load(path="config.ini"):
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg


def strs(cfg, section, key):
    return [x.strip() for x in cfg.get(section, key).split(",") if x.strip()]


def ints(cfg, section, key):
    return [int(x) for x in strs(cfg, section, key)]


def mapping(cfg, section, key):
    return {k: int(v) for k, v in (x.split(":") for x in strs(cfg, section, key))}


def path(cfg, key):
    return cfg.get("paths", key)


def data_file(cfg, name):
    return os.path.join(path(cfg, "data_dir"), f"{name}.npz")


def results_dir(cfg, model, dataset):
    d = os.path.join(path(cfg, "outputs_dir"), "results", model, dataset)
    os.makedirs(d, exist_ok=True)
    return d
