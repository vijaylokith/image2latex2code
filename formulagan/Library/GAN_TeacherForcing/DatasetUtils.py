"""
Dataset Utils for IM2LATEX_100K Dataset

Expected Files in Dataset Folder:
    - formula_images_processed/formula_images_processed/    :  All images of codes are saved in this folder
    - im2latex_formulas.norm.csv                            :  A list of all codes only
    - im2latex_train.csv                                    :  Code to Image Path Mappings for Train are saved in this file
    - im2latex_validate.csv                                 :  Code to Image Path Mappings for Validation are saved in this file
    - im2latex_test.csv                                     :  Code to Image Path Mappings for Test are saved in this file
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
DATASET_PATH = "Data/Datasets/IM2LATEX_100K/IM2LATEX_100K/"
DATASET_ITEMPATHS = {
    "images": "formula_images_processed/formula_images_processed/",
    "codes": "im2latex_formulas.norm.csv",
    "train": "im2latex_train.csv",
    "val": "im2latex_validate.csv",
    "test": "im2latex_test.csv"
}

# Main Functions
# Load Functions
def DatasetUtils_LoadCSV(path):
    '''
    DatasetUtils - Load CSV
    '''
    return pd.read_csv(path)

# Extract Functions
def DatasetUtils_ExtractZIPDataset(path, save_path=None, **params):
    '''
    DatasetUtils - Extract the Zipped Dataset
    '''
    # Set same dir if save_path is not specified
    if save_path is None:
        save_path = os.path.dirname(path)
    # Extract
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(save_path)

# Conversion Functions
def DatasetUtils_LaTeX2Image(code, euler=False, dpi=1000, **params):
    '''
    DatasetUtils - LaTeX code to Image
    '''
    code = "$$" + code + "$$"
    out = io.BytesIO()
    sympy.preview(code, euler=euler, output="png", viewer="BytesIO", outputbuffer=out, dvioptions=["-D", str(dpi)], **params)
    I = Image.open(out)
    I = np.array(I.convert("RGB"))

    return I

# Dataset Functions
def DatasetUtils_LoadDatasetFull(path=DATASET_PATH, N=-1, **params):
    '''
    DatasetUtils - Load IM2LATEX_100K Dataset - Full
    Pandas Dataframe with columns:
        - "code" - LaTeX code : Ground Truth Y
        - "path" - Path to image : Input X
    '''
    # Load Datasets
    dataset_train = DatasetUtils_LoadDataset(path, mode="train", N=N, **params)
    dataset_val = DatasetUtils_LoadDataset(path, mode="val", N=N, **params)
    dataset_test = DatasetUtils_LoadDataset(path, mode="test", N=N, **params)
    # Combine Datasets
    dataset_full = pd.concat([dataset_train, dataset_val, dataset_test], ignore_index=True)

    return dataset_full

def DatasetUtils_LoadDataset(path=DATASET_PATH, mode="train", N=-1, **params):
    '''
    DatasetUtils - Load IM2LATEX_100K Dataset
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

# Main Vars
DATASET_FUNCS = {
    "full": DatasetUtils_LoadDatasetFull,
    "train": functools.partial(DatasetUtils_LoadDataset, mode="train"),
    "val": functools.partial(DatasetUtils_LoadDataset, mode="val"),
    "test": functools.partial(DatasetUtils_LoadDataset, mode="test")
}