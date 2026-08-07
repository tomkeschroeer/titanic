import os
from typing import Optional
import yaml
import pandas as pd
import numpy as np

class ConfigLoader: 
    """
    A class to load and validate configuration files in YAML format.
    """
    def __init__(
        self, 
        config_path: Optional[str] = None
    ):
        """
        initialise the config and define the required and optional parameters

        Parameters:
        -----------
        config_path : Optional[str] = None
            path to config file
        """
        # check if file is passed and exists. Raise error otherwise
        if config_path is None:
            raise TypeError(f"No config file passed")
        elif not os.path.exists(config_path):
            raise ValueError(f"config file {config_path} does not exist")

        with open(config_path, 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error parsing YAML file: {exc}")

        # Define which values are expected to be in the config file and which are optional
        self.required_values = [
            "model", 
            "output_path",
            "train_file",
            "test_file",
            "solution_file"
        ]

        # default values for optional values that are not model specific
        self.default_optional_values_gen = {
            "input_feats": ["Sex","Pclass", "Age","SibSp","Parch"],
            "label": "Survived",
            "postfix": "",
            "eval": {},
        }

        # default values for optional values that are model specific
        self.default_optional_values = {
            "gbt": {
                "lr": 0.05,
                "n_estimators": 500,
                "num_leaves" : 31,
            },
            "xgb": {
                "learning_rate": 0.3, 
                "max_depth": 6, 
            },
        }

    def load_config(self):
        """
        load values defined in config file as attributes
        """
        for val in self.required_values:
            if val not in self.config: raise ValueError(f"value {val} not found but manatory.")
            setattr(self, val, self.config.get(val))

        model_type = self.model["type"]
        del self.model["type"]
        self.type = model_type

        for val in self.default_optional_values_gen.keys():
            if val not in self.config:
                print(f"WARNING: Configuration for {val} not set. Setting to default value {self.default_optional_values_gen.get(val, None)}")
            setattr(self, val, self.config.get(
                val, 
                self.default_optional_values_gen.get(val, None))
            )
        for val in self.default_optional_values[model_type].keys():
            if val not in self.config["model"]:
                print(f"WARNING: Configuration for {val} not set. Setting to default value {self.default_optional_values[model_type].get(val, None)}")
            self.model[val] = self.config["model"].get(
                val,
                self.default_optional_values[model_type].get(val, None)
            )

class DataLoader:
    """
    A class to load and preprocess data for forecasting tasks.
    """
     
    def __init__(
        self, 
        file_name: Optional[str] = None, 
        input_feats: list = [],
        label: Optional[str] = None,
        testing: bool = False,
        dropnans: bool = False
    ):
        """
        Initialise dataloader

        Parameters:
        -----------
        file_name : str, optional
            The file name containing the data. Defaults to None.
        input_feats: list, optional
            List of the input features to be used during the training
        label: str, optional
            feature that should be predicted
        """

        self.file_name = file_name
        self.input_feats = input_feats
        self.label = label
        self.testing = testing
        self.dropnans = dropnans

        # ensure label is not added as an input feature
        if self.label in self.input_feats:
            raise ValueError("label were added to input features. Please remove the label from the input features and try again.") 
    
    def define_inputs(self):
        mask_female =  self.df["Sex"] == "female"
        mask_male = self.df["Sex"] == "male"
        self.df["Sex"] = np.select([mask_female, mask_male], [1,0])

    def load(self):
        """
        Load input files. The content of the input files is loaded into pandas dataframes 
        and split into a train and a test set. It is ensured that no future data is used when
        predicting a future month.
        """

        # Read and combine the CSV files into a DataFrame
        self.df = pd.read_csv(self.file_name)

        self.define_inputs()

        # for the baseline approach these parameters are needed. They will not be used in the training of the models
        if self.testing:
            self.data = self.df[self.input_feats + ["PassengerId"]]
        else:
            self.data = self.df[self.input_feats + [self.label]]

        if self.dropnans:
            self.data = self.data.dropna()
        
        if self.testing:
            return self.data, None
        
        self.data_train = self.data[self.input_feats]
        self.data_labels = self.data[self.label]

        return self.data_train, self.data_labels

