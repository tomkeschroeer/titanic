import argparse
import os
import joblib

import lightgbm as lgb

from src.tools import (
    ConfigLoader, 
    DataLoader,
)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="path to config file", required=True)
args = parser.parse_args()
config_path = args.config

config = ConfigLoader(config_path)
config.load_config()

data_loader = DataLoader(
    file_name = config.train_file,
    input_feats = config.input_feats,
    label = config.label
)

train_data, train_labels = data_loader.load()

os.makedirs(config.output_path, exist_ok=True)

model_output_dir = config.output_path + f"/{config.type}".replace("//","/")
os.makedirs(model_output_dir, exist_ok=True)

if config.type == "gbt":
    model = lgb.LGBMClassifier(
        learning_rate=config.lr,
        num_leaves=config.num_leaves,
        n_estimators=config.n_estimators
    )
    model.fit(train_data, train_labels)
    joblib.dump(model, f"{model_output_dir}/model.joblib")