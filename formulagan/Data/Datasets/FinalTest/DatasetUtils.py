"""
Dataset Utils for Final Test Dataset
"""

# Imports
import io
import os
import cv2
import sympy
import zipfile
import functools
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Main Vars
DATASET_PATH = "Data/Datasets/FinalTest/TestData/"
DATASET_ITEMPATHS = {
    "images": "",
    "test": "",
}

# Main Functions
# Load Functions
def DatasetUtils_LoadCSV(path):
    '''
    DatasetUtils - Load CSV
    '''
    return pd.read_csv(path)

# Dataset Functions
def DatasetUtils_LoadDataset(path=DATASET_PATH, mode="test_short", N=-1, DATASET_ITEMPATHS=DATASET_ITEMPATHS, **params):
    '''
    DatasetUtils - Load Final Test Dataset
    Pandas Dataframe with columns:
        - "code" - LaTeX code : Ground Truth Y
        - "path" - Path to image : Input X
    '''
    # Get Dataset Labels
    dataset_info = DatasetUtils_LoadCSV(os.path.join(path, DATASET_ITEMPATHS[mode]))
    # Take N range
    if type(N) == int:
        if N > 0: dataset_info = dataset_info.head(N)
    elif type(N) == list:
        if len(N) == 2: dataset_info = dataset_info.iloc[N[0]:N[1]]
    # Reset Columns
    dataset_info.columns = ["code", "path"]
    # Add Main Path
    dataset_info["path"] = dataset_info["path"].apply(lambda x: os.path.join(path, DATASET_ITEMPATHS["images"], x))

    return dataset_info