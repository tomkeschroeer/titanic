import numpy as np
import os
import argparse
import joblib
import pandas as pd
import csv

from src.tools import (
    ConfigLoader, 
    DataLoader,
)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="path to config file", required=True)
args = parser.parse_args()

config = ConfigLoader(args.config)
config.load_config()

outpath = f"{config.output_path}/{config.type}{config.postfix}"
if not os.path.exists(outpath):
    raise ValueError(f"No training folder available for training type {config.type}. Looked here: {outpath}")

dataloader = DataLoader( 
        file_name=config.test_file,
        input_feats=config.input_feats,
        label=config.label,
        testing=True
    )

# only need test set, no labels, to get predictions
test_set, _ = dataloader.load()

output_path_plots = f"{outpath}/plots"
os.makedirs(output_path_plots, exist_ok=True)

model = joblib.load(f"{outpath}/model.joblib")
preds = model.predict(test_set[config.input_feats])
pass_ids = test_set["PassengerId"]

sol_columns = ["PassengerId","Survived"]
sol = pd.DataFrame(np.stack((pass_ids,preds),axis=1), columns = ["PassengerId","Survived"])
sol.to_csv(f"{outpath}/sol.csv", columns=sol_columns, index=False)