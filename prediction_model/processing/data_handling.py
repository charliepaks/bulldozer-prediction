import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath, low_memory=False, parse_dates=config.SALE_DATE)# load data and parse dates
    _data.columns = [c.strip() for c in _data.columns] # Fix Column names
    _data = _data.sort_values(by=[config.SALE_DATE], inplace=True, ascending=True)
    _data_tmp = _data.copy()
    _data_tmp[config.SALE_YEAR]= _data_tmp.saledate.dt.year
    _data_tmp[config.SALE_MONTH] = _data_tmp.saledate.dt.month
    _data_tmp[config.SALE_DAY] = _data_tmp.saledate.dt.day
    _data_tmp[config.SALE_DAY_OF_WEEK] = _data_tmp.saledate.dt.dayofweek
    _data_tmp[config.SALE_DAY_OF_YEAR] = _data_tmp.saledate.dt.dayofyear
    _data_tmp.drop(config.SALE_DATE, axis=1, inplace=True)
    # Convert strings to categories
    for label, content in _data_tmp.items():
        if pd.api.types.is_object_dtype(content):
            _data_tmp[label] = content.astype("category").cat.as_ordered()

    # Fill missing numeric values
    for label, content in _data_tmp.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                _data_tmp[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                _data_tmp[label] = content.fillna(content.median())
    # Turn categorical variables into numbers and fill missing
    for label, content in _data_tmp.items():
        if not pd.api.types.is_numeric_dtype(content):
            # Add binary column to indicate wheter sample has missing value
            _data_tmp[label+"_is_missing"] = pd.isnull(content)
            # Turn categories into numbers and add +1
            _data_tmp[label] = pd.Categorical(content).codes+1

    return _data_tmp
    

# Separate X and y
def separate_data(data):
    X = data.drop(config.TARGET, axis=1)
    y= data[config.TARGET]
    return X,y

#Split the dataset
def split_data(X, y, test_size=0.2, random_state=42):
  # Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  return X_train, X_test, y_train, y_test

#Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    print(save_path)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

#Deserialization
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH,pipeline_to_load)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded