"""
Streamlit App - Model
"""

# Imports
import os
import time
import json
import streamlit as st

from Utils.ModelVis import *

from Library.GAN_TeacherForcing.ModelLibrary import Encoder as GAN_TeacherForcing_Encoder
from Library.GAN_TeacherForcing.ModelLibrary import Decoder as GAN_TeacherForcing_Decoder
from Library.GAN_TeacherForcing.ModelLibrary import Discriminator as GAN_TeacherForcing_Discriminator

# Set Verbose to False
for module in [GAN_TeacherForcing_Encoder, GAN_TeacherForcing_Decoder, GAN_TeacherForcing_Discriminator]:
    for k in module.VERBOSE.keys():
        module.VERBOSE[k] = False

# Main Vars
MODEL_PATHS = {
    "Encoder-Decoder Simple": "Models/FormulaGAN/EncDec_Simple_Best.h5",
    "Encoder-Decoder Recurrent": "Models/FormulaGAN/EncDec_Recurrent_Best.h5",
    "Encoder-Decoder Attention": "Models/FormulaGAN/EncDec_Attention_Best.h5",
    "GAN Simple": "Models/FormulaGAN/GAN_Simple_Best.h5",
}
TEMP_PATH = "Data/Temp/"

# UI Functions
def UI_LoadModel():
    '''
    Load Model
    '''
    st.markdown("## Load Model")
    # Select Model
    USERINPUT_Model = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
    # Load Model
    MODEL = Model_LoadModel(MODEL_PATHS[USERINPUT_Model])

    return MODEL

# Model Part Generate UI Functions
def ModelPartUI_Encoder():
    '''
    FormulaGAN Image Encoder Generate
    '''
    encoder_params_default = {
        "conv_n_filters": [16, 32],
        "conv_activation": "tanh",
        "conv_dropout": 0.2,
        "dense_n_units": [16],
        "dense_activation": "tanh",
        "dense_dropout": 0.1, 
        "output_activation": "tanh",
        "output_shape": 32
    }

    st.markdown("## Encoder")
    encoder_type = st.selectbox("Select Encoder Type", list(GAN_TeacherForcing_Encoder.ENCODERS.keys()))
    encoder_params_str = st.text_area(
        "Encoder Params (JSON Format)", str(json.dumps(encoder_params_default, indent=4)), 
        height=200
    )
    encoder_params = {
        "type": encoder_type,
        **(dict(json.loads(encoder_params_str)))
    }

    return encoder_params

def ModelPartUI_Decoder(encoder_output_shape=512, max_sequence_length=100):
    '''
    FormulaGAN Decoder Generate
    '''
    decoder_params_optional = {
        "token_output": {
            "simple": {
                "decoder_input_dense_units": [32],
                "decoder_input_dense_activation": "tanh",
                "decoder_input_dense_dropout": 0.0,
                "decoder_output_dense_units": [32, 64],
                "decoder_output_dense_activation": "tanh",
                "decoder_output_dense_dropout": 0.0,
            },
            "recurrent": {
                "decoder_input_recurrent_dense_units": 32,
                "decoder_input_recurrent_dense_activation": "tanh",
                "decoder_output_recurrent_units": [32, 64],
                "decoder_output_recurrent_dense_units": [32],
                "decoder_output_recurrent_dense_activation": None,
                "decoder_output_dense_units": [64, 128],
                "decoder_output_dense_activation": "tanh",
                "decoder_output_dense_dropout": 0.0,
            },
        },
        "sequence_output": {
            "recurrent": {
                "decoder_input_recurrent_dense_units": 32,
                "decoder_input_recurrent_dense_activation": "tanh",
                "decoder_output_recurrent_units": [32, 64],
                "decoder_output_recurrent_dense_units": [32],
                "decoder_output_recurrent_dense_activation": None,
                "decoder_output_dense_units": [64, 128],
                "decoder_output_dense_activation": "tanh",
                "decoder_output_dense_dropout": 0.0,
            },
        }
    }
    decoder_params_default = {
        "decoder_input_embedding_dim": 32,
        "decoder_input_recurrent_units": 32,
    }
    concat_shape_params = {
        "encoder_output_shape": encoder_output_shape,
        "max_sequence_length": max_sequence_length
    }

    st.markdown("## Decoder")
    decoder_outputtype = st.selectbox("Select Decoder Output Type", list(GAN_TeacherForcing_Decoder.DECODERS.keys()))
    decoder_type = st.selectbox("Select Decoder Type", list(GAN_TeacherForcing_Decoder.DECODERS[decoder_outputtype].keys()))
    decoder_params_default.update(decoder_params_optional[decoder_outputtype][decoder_type])
    decoder_params_str = st.text_area(
        "Decoder Params (JSON Format)", str(json.dumps(decoder_params_default, indent=4)), 
        height=200
    )
    decoder_params_dict = dict(json.loads(decoder_params_str))

    decoder_params = {
        "output_type": decoder_outputtype,
        "type": decoder_type,
        "concat_layer_shape": GAN_TeacherForcing_Decoder.DECODERS_CONCAT_SHAPE[decoder_outputtype][decoder_type](decoder_params_dict, concat_shape_params),
        **(decoder_params_dict)
    }

    return decoder_params

def ModelPartUI_Discriminator():
    '''
    FormulaGAN Discriminator Generate
    '''
    discriminator_params_optional = {
        "token": {
            "token_rep_params": {
                "type": "dense", 
                "token_dense_units": [16, 8],
                "token_dense_activation": "tanh",
                "token_dense_dropout": 0.0
            }, 
            "dense_n_units": [16, 8],
            "dense_activation": "tanh",
            "dense_dropout": 0.1, 
        }
    }
    discriminator_params_default = {}

    st.markdown("## Discriminator")
    discriminator_type = st.selectbox("Select Discriminator Type", list(GAN_TeacherForcing_Discriminator.DISCRIMINATORS.keys()))
    discriminator_params_default.update(discriminator_params_optional[discriminator_type])
    discriminator_params_str = st.text_area(
        "Discriminator Params (JSON Format)", str(json.dumps(discriminator_params_default, indent=4)), 
        height=200    
    )
    discriminator_params = {
        "type": discriminator_type,
        **(dict(json.loads(discriminator_params_str)))
    }

    return discriminator_params

# Model Generate UI Functions
def ModelUI_EncoderDecoder():
    '''
    FormulaGAN Encoder-Decoder Model Generate
    '''
    # Title
    st.markdown("# Encoder-Decoder")
    # Init Model
    # Params
    cols = st.columns(3)
    image_size = (
        cols[0].number_input("Image Height", min_value=1, value=54, step=1),
        cols[1].number_input("Image Width", min_value=1, value=256, step=1),
        cols[2].number_input("Image Channels", min_value=1, value=1, step=1)
    )
    cols = st.columns(2)
    vocab_size = cols[0].number_input("Vocab Size", min_value=1, value=567, step=1)
    max_sequence_length = cols[1].number_input("Max Sequence Length", min_value=1, value=100, step=1)
    encoder_params = ModelPartUI_Encoder()
    decoder_params = ModelPartUI_Decoder(
        encoder_output_shape=encoder_params["output_shape"], max_sequence_length=max_sequence_length
    )

    # Generate Model
    N_Steps = 3
    progressObj = st.progress(0/N_Steps)
    # Encoder
    encoder = GAN_TeacherForcing_Encoder.ENCODERS[encoder_params["type"]](image_size, **encoder_params)
    progressObj.progress(1/N_Steps)
    # Decoder
    decoder = GAN_TeacherForcing_Decoder.DECODERS[decoder_params["output_type"]][decoder_params["type"]](
        encoder["output"],
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        **decoder_params
    )
    progressObj.progress(2/N_Steps)
    # Form Model
    model = GAN_TeacherForcing_Encoder.Model([encoder["input"], decoder["input"]], decoder["output"])
    progressObj.progress(3/N_Steps)

    models = {
        "Encoder-Decoder Model": model
    }
    return models

def ModelUI_GAN(decoder_concat_name="decoder_concat"):
    '''
    FormulaGAN GAN Model Generate
    '''
    # Title
    st.markdown("# GAN")
    # Init Model
    # Params
    cols = st.columns(3)
    image_size = (
        cols[0].number_input("Image Height", min_value=1, value=54, step=1),
        cols[1].number_input("Image Width", min_value=1, value=256, step=1),
        cols[2].number_input("Image Channels", min_value=1, value=1, step=1)
    )
    cols = st.columns(2)
    vocab_size = cols[0].number_input("Vocab Size", min_value=1, value=567, step=1)
    max_sequence_length = cols[1].number_input("Max Sequence Length", min_value=1, value=100, step=1)
    encoder_params = ModelPartUI_Encoder()
    decoder_params = ModelPartUI_Decoder(
        encoder_output_shape=encoder_params["output_shape"], max_sequence_length=max_sequence_length
    )
    discriminator_params = ModelPartUI_Discriminator()
    discriminator_params.update({
        "datarep_dim": decoder_params["concat_layer_shape"]
    })

    # Generate Model
    N_Steps = 4
    progressObj = st.progress(0/N_Steps)
    # Encoder
    encoder = GAN_TeacherForcing_Encoder.ENCODERS[encoder_params["type"]](image_size, **encoder_params)
    progressObj.progress(1/N_Steps)
    # Decoder
    decoder = GAN_TeacherForcing_Decoder.DECODERS[decoder_params["output_type"]][decoder_params["type"]](
        encoder["output"],
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        **decoder_params
    )
    progressObj.progress(2/N_Steps)
    # Form Generator
    generator = GAN_TeacherForcing_Encoder.Model([encoder["input"], decoder["input"]], decoder["output"])
    progressObj.progress(3/N_Steps)
    # Form Discriminator
    discriminator = GAN_TeacherForcing_Discriminator.DISCRIMINATORS[discriminator_params["type"]](
        vocab_size=vocab_size, **discriminator_params
    )
    progressObj.progress(4/N_Steps)

    models = {
        "Generator": generator,
        "Discriminator": discriminator
    }
    return models

def UI_ModelVis(MODEL):
    '''
    Model Visualisations
    '''
    # Visualisations
    # Keras
    st.markdown("## Model Vis - Keras")
    try:
        save_path = os.path.join(TEMP_PATH, "ModelVis_Keras_BlockView.png")
        I_Keras = ModelVis_Keras_SequentialModel_BlockView(
            MODEL, save_path=save_path, dpi=96, display=False, 
            show_shapes=True, show_dtype=False, show_layer_names=True, show_layer_activations=True,
            expand_nested=True
        )
        st.image(I_Keras, caption="Keras Block View", use_column_width=True)
    except Exception as e:
        print(e)
        st.error("Keras Block View not working")
    # Keras Visualiser
    st.markdown("## Model Vis - Keras Visualiser")
    try:
        save_path = os.path.join(TEMP_PATH, "ModelVis_KerasVisualiser.png")
        I_KerasVisualiser = ModelVis_KerasVisualizer_SequentialModel(MODEL, save_path=save_path, display=False)
        st.image(I_KerasVisualiser, caption="Keras Visualiser", use_column_width=True)
    except Exception as e:
        print(e)
        st.error("Keras Visualiser not working")
    # VisualKeras
    st.markdown("## Model Vis - Visualkeras")
    try:
        save_path = os.path.join(TEMP_PATH, "ModelVis_VisualKeras_LayerView.png")
        I_VisualKeras_LayerView = ModelVis_VisualKeras_SequentialModel_LayerView(MODEL, save_path=save_path, display=False)
        st.image(I_VisualKeras_LayerView, caption="VisualKeras Layer View", use_column_width=True)
    except Exception as e:
        print(e)
        st.error("VisualKeras Layer View not working")
    try:
        save_path = os.path.join(TEMP_PATH, "ModelVis_VisualKeras_GraphView.png")
        I_VisualKeras_GraphView = ModelVis_VisualKeras_SequentialModel_GraphView(MODEL, save_path=save_path, display=False)
        st.image(I_VisualKeras_GraphView, caption="VisualKeras Graph View", use_column_width=True)
    except Exception as e:
        print(e)
        st.error("VisualKeras Graph View not working")

# Model Type Vars
FORMULAGAN_MODEL_TYPES = {
    "Encoder-Decoder": ModelUI_EncoderDecoder,
    "GAN": ModelUI_GAN,
}

# Main Functions
def model_visualise():
    # Title
    st.markdown("# Model Visualisation")

    # Load Model
    MODEL = UI_LoadModel()
    # Visualise
    UI_ModelVis(MODEL)

def formula_gan_model_generate():
    # Title
    st.markdown("# Formula GAN Model Generator")

    # Select Model Type
    USERINPUT_model_type = st.selectbox("Select Model Type", list(FORMULAGAN_MODEL_TYPES.keys()))
    # Generate Models
    MODELS = FORMULAGAN_MODEL_TYPES[USERINPUT_model_type]()
    # Visualise
    for k in MODELS.keys():
        st.markdown("# " + k)
        UI_ModelVis(MODELS[k])

# Mode Vars
MODES = {
    "Model Visualise": model_visualise,
    "FormulaGAN Model Generate": formula_gan_model_generate
}

# App Functions
def app_main():
    # Title
    st.markdown("# Image to LaTeX Model Utils")
    # Mode
    USERINPUT_Mode = st.sidebar.selectbox(
        "Select Mode",
        list(MODES.keys())
    )
    MODES[USERINPUT_Mode]()

# RunCode
if __name__ == "__main__":
    # Assign Objects
    
    # Run Main
    app_main()