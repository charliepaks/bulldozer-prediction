from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import os
import sys
from prediction_model.config import config

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import numpy as np

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        # Fit the scaler to the training data
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        # Apply the scaler to the data
        X_scaled = self.scaler.transform(X)
        return X_scaled
