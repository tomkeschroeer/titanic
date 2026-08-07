import argparse
import os
import joblib
import shutil

import lightgbm as lgb
from xgboost import XGBClassifier

from src.tools import (
    ConfigLoader, 
    DataLoader,
)

from src.plotting_tools import (
    plot_inputs
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

model_output_dir = config.output_path + f"/{config.type}" + f"{config.postfix}".replace("//","/") 
os.makedirs(model_output_dir, exist_ok=True)
shutil.copyfile(config_path, f"{model_output_dir}/config.yaml")

if config.eval.get("plot_inputs",False):
    plot_inputs(train_data, train_labels, config.input_feats,model_output_dir)

if config.type == "gbt":
    model = lgb.LGBMClassifier(**config.model)
    model.fit(train_data, train_labels)
    joblib.dump(model, f"{model_output_dir}/model.joblib")

if config.type == "xgb":
    model = XGBClassifier(**config.model)
    model.fit(train_data, train_labels)
    joblib.dump(model, f"{model_output_dir}/model.joblib")
