"""
Dataset Utils for Generated Datasets

Expected Files in Dataset Folder:
    - Images/    :  All images of codes are saved in this folder
    - labels.csv :  Code to Image Path Mappings are saved in this file
"""

# Imports
import io
import os
import cv2
import sympy
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Main Vars
DATASET_PATH = "Data/Datasets/GeneratedDatasets/Test/"
DATASET_ITEMPATHS = {
    "images": "Images/",
    "labels": "labels.csv"
}

# Main Functions
# Load Functions
def DatasetUtils_LoadCSV(path):
    '''
    Load CSV
    '''
    return pd.read_csv(path)

# Conversion Functions
def DatasetUtils_LaTeX2Image(code, euler=False, dpi=1000, **params):
    '''
    LaTeX code to Image
    '''
    code = "$$" + code + "$$"
    out = io.BytesIO()
    sympy.preview(code, euler=euler, output="png", viewer="BytesIO", outputbuffer=out, dvioptions=["-D", str(dpi)], **params)
    I = Image.open(out)
    I = np.array(I.convert("RGB"))

    return I

# Dataset Read Functions
def DatasetUtils_ReadCodes(path, codes_col="code", **params):
    '''
    Read Codes from TXT/CSV File
    '''
    codes = []
    # Read
    ext = os.path.splitext(path)[1]
    if ext in [".txt"]:
        with open(path, "r") as f:
            codes = f.read().split("\n")
    elif ext in [".csv"]:
        codes = pd.read_csv(path)[codes_col].values
    
    return codes

# Dataset Generation Functions
def DatasetUtils_GenerateLaTeXImageDataset(codes, path, progress=True, **params):
    '''
    Generate LaTeX Image Dataset

    Structure of Saved Data:
    path/
        - Images/    :  All images of codes are saved in this folder with name as id of code in labels.csv
        - labels.csv :  Labels / ID of codes are saved in this file
    '''
    # Create Folders
    if not os.path.exists(os.path.join(path, DATASET_ITEMPATHS["images"])):
        os.mkdir(os.path.join(path, DATASET_ITEMPATHS["images"]))
    # Generate Dataset
    labels = {
        "id": [],
        "code": [],
        "path": []
    }
    for i in tqdm(range(len(codes)), disable=not progress):
        code = codes[i]
        labels["id"].append(i)
        labels["code"].append(code)
        labels["path"].append(str(i) + ".png")
        I = DatasetUtils_LaTeX2Image(code, **params)
        cv2.imwrite(os.path.join(path, DATASET_ITEMPATHS["images"], str(i) + ".png"), I)
        # Update Progress
        if "progressObj" in params.keys() and params["progressObj"] is not None:
            params["progressObj"].progress((i+1) / len(codes))
    # Save Labels
    labels = pd.DataFrame.from_dict(labels)
    labels.to_csv(os.path.join(path, DATASET_ITEMPATHS["labels"]), index=False)

# Dataset Functions
def DatasetUtils_LoadDataset(path=DATASET_PATH, **params):
    '''
    Load GeneratedDatasets Dataset
    Pandas Dataframe with columns:
        - "code" - LaTeX code : Ground Truth Y
        - "path" - Path to image : Input X
    '''
    # Get Dataset Labels
    dataset_info = DatasetUtils_LoadCSV(os.path.join(path, DATASET_ITEMPATHS["labels"]))
    # Add Main Path
    dataset_info["path"] = dataset_info["path"].apply(lambda x: os.path.join(path, DATASET_ITEMPATHS["images"], x))
    # Drop ID Column
    dataset_info = dataset_info.drop(columns=["id"])
    
    return dataset_info

# Main Vars
DATASET_FUNCS = {
    "full": DatasetUtils_LoadDataset,
    "train": DatasetUtils_LoadDataset,
    "val": DatasetUtils_LoadDataset,
    "test": DatasetUtils_LoadDataset
}