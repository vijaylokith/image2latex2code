"""
Dataset Utils for Final Test Dataset - Short

Expected Files in Dataset Folder:
    - Images_Short/                 :  All images of codes are saved in this folder
    - finaltest_short_test.csv      :  Code to Image Path Mappings for Test are saved in this file
"""

# Imports
from .DatasetUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/FinalTest/TestData/"
DATASET_ITEMPATHS = {
    "images": "Images_Short/",
    "test": "finaltest_short_test.csv"
}

# Main Vars
DATASET_FUNCS = {
    "full": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "train": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "val": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "test": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS)
}