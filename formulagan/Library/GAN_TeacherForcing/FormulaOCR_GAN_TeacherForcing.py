"""
Formula OCR

Encoder-Decoder Architecture like Image Captioning with GAN and Teacher Forcing
Docs: 
https://medium.com/analytics-vidhya/image-captioning-with-tensorflow-2d72a1d9ffea
https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
"""

# Imports
from .Model import *
# import streamlit as st

# Main Vars
OCR_PATHS = {
    "model": "Models/FormulaGAN/EncDec_Attention_Best.h5",
    "tokenizer": "Library/GAN_TeacherForcing/Models/tokenizer_full.tf",
}
# Model Loading and Init
OCR = {
    "model": None,#Model_LoadModel(OCR_PATHS["model"]),
    "tokenizer": Dataset_Tokenizer_LoadTokenizer(OCR_PATHS["tokenizer"])
}
OCR["lookups"] = Dataset_Tokenizer_GetLookups(OCR["tokenizer"])

# Conversion Functions
def ImageUtils_Bytes2TF(I_bytes):
    '''
    Image Bytes to Tensorflow Image
    '''
    # Load Image
    I_array = np.array(Image.open(io.BytesIO(I_bytes)), dtype=float)
    print("GAN 1:", I_array.shape, I_array.dtype, I_array.min(), I_array.max())
    # Greyscale
    if I_array.ndim == 3: I_array = np.mean(I_array[:, :, :3], axis=-1)
    # Normalize
    if I_array.max() > 1.0: I_array /= 255.0
    # Check Background
    bg_white = (1.0 - I_array.mean()) < (I_array.mean() - 0.0)
    print("GAN 2:", I_array.shape, I_array.dtype, I_array.min(), I_array.max())

    # Convert
    I = I_array
    I = tf.convert_to_tensor(I, dtype=tf.float32)
    if I.ndim == 2: I = tf.expand_dims(I, axis=-1)
    # Apply Padding
    padding_val = 1.0 if bg_white else 0.0
    I = Image_ResizeWithPadding(I, DATASET_PARAMS["images"]["size"], padding_val=padding_val)
    # Normalize
    I_array = I.numpy()
    I = (I - I_array.min()) / (I_array.max() - I_array.min())

    I_array = I.numpy()
    print("GAN 3:", I_array.shape, I_array.dtype, I_array.min(), I_array.max())
    # st.image(I_array, caption=f"GAN Input Image ({I_array.shape[0]}, {I_array.shape[1]})", use_column_width=True)

    return I

# Main Functions
def FormulaOCR_ConvertImage2Latex(I_bytes, partial_code="", **params):
    '''
    Image Bytes to Latex Code using FormulaGAN Module
    Optional Params:
        - partial_code: Partial Code Text to Start with
    '''
    global OCR

    # Convert Image Bytes to Tensorflow Image
    I = ImageUtils_Bytes2TF(I_bytes)
    # Set Partial Starting Code
    DECODER_INPUT_TEXT = "<start>"
    if not partial_code.strip() == "":
        DECODER_INPUT_TEXT += " " + partial_code
    # Predict using Model
    pred_seq = Model_Predict(
        OCR["model"], OCR["tokenizer"], OCR["lookups"], 
        I=I, DECODER_INPUT_TEXT=DECODER_INPUT_TEXT
    )
    # Clean LaTeX Code
    latex_code = pred_seq.replace("<start>", "").replace("<end>", "").strip()
    latex_code = [latex_code]

    return latex_code

# Main Vars
PIX2TEX_FUNCS = {
    "convert_image_to_latex": FormulaOCR_ConvertImage2Latex,
    "tokenizer": OCR["tokenizer"],
    "lookups": OCR["lookups"]
}

# RunCode