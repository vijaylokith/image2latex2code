"""
Utils Imports and Common Functions
"""

# Imports
# General Imports
import io
import os
import json
import time
import functools
from tqdm.notebook import tqdm
# General ML Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Keras Model Imports
from keras.layers import *
from keras.models import load_model, Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy
# Keras Utils Imports
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

# Print Functions
def PRINT_FN(print_module, *args, **kwargs):
    if VERBOSE[print_module]:
        print(*args, **kwargs)

# Main Vars
VERBOSE = {
    "encoder": True,
    "decoder": True,
    "discriminator": True,
    "token_rep": True,
    "token_combine": True
}
PRINT_FUNCS = {
    "encoder": functools.partial(PRINT_FN, "encoder"),
    "decoder": functools.partial(PRINT_FN, "decoder"),
    "discriminator": functools.partial(PRINT_FN, "discriminator"),
    "token_rep": functools.partial(PRINT_FN, "token_rep"),
    "token_combine": functools.partial(PRINT_FN, "token_combine")
}

# Main Functions