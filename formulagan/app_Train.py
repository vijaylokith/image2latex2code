"""
Streamlit App - Train
"""

# Imports
import os
import time
import json
import streamlit as st

from Library.GAN_TeacherForcing.Model import *
for k in VERBOSE.keys(): VERBOSE[k] = False

# Main Vars
PATHS = {
    "params": "Library/GAN_TeacherForcing/params.json",
    "temp": "Data/Temp/"
}

# UI Functions
def UI_LoadEncoderParams(PARAMS_JSON):
    '''
    Load Encoder Params
    '''
    col1, col2 = st.columns((1, 3))
    encoder_types = list(ENCODERS.keys())
    encoder_params = {
        "type": col1.selectbox("Encoder Type", encoder_types, encoder_types.index(PARAMS_JSON["encoder_params"]["type"]))
    }
    encoder_params_default = {
        **PARAMS_JSON["encoder_params"]["common"],
        **PARAMS_JSON["encoder_params"][encoder_params["type"]]
    }
    encoder_params_str = col2.text_area("Encoder Params", json.dumps(encoder_params_default, indent=4), height=300)
    encoder_params.update(json.loads(encoder_params_str))

    return encoder_params

def UI_LoadDecoderParams(PARAMS_JSON):
    '''
    Load Decoder Params
    '''
    col1, col2 = st.columns((1, 3))
    decoder_types = list(DECODERS["token_output"].keys())
    decoder_params = {
        "type": col1.selectbox("Decoder Type", decoder_types, decoder_types.index(PARAMS_JSON["decoder_params"]["type"]))
    }
    decoder_params_default = {
        **PARAMS_JSON["decoder_params"]["common"],
        **PARAMS_JSON["decoder_params"][decoder_params["type"]]
    }
    decoder_params_str = col2.text_area("Decoder Params", json.dumps(decoder_params_default, indent=4), height=300)
    decoder_params.update(json.loads(decoder_params_str))

    return decoder_params

def UI_LoadDiscriminatorParams(PARAMS_JSON, encoder_params, decoder_params, MAX_SEQUENCE_LENGTH):
    '''
    Load Discriminator Params
    '''
    # Token Rep Params
    token_rep_params_str = st.text_area("Token Rep Params", json.dumps(PARAMS_JSON["token_rep_params"], indent=4), height=300)
    token_rep_params = json.loads(token_rep_params_str)
    concat_shape_params = {
        "encoder_output_shape": encoder_params["output_shape"],
        "max_sequence_length": MAX_SEQUENCE_LENGTH
    }
    # Discriminator Params
    discriminator_params = {
        "datarep_dim": DECODERS_CONCAT_SHAPE["token_output"][decoder_params["type"]](decoder_params, concat_shape_params), 
        "token_rep_params": token_rep_params
    }
    discriminator_params_str = st.text_area("Discriminator Params", json.dumps(PARAMS_JSON["discriminator_params"], indent=4), height=300)
    discriminator_params.update(json.loads(discriminator_params_str))
    # Discriminator Compile Params
    discriminator_compile_params = {
        "loss_fn": BinaryCrossentropy(),
        "optimizer": Adam(learning_rate=float(st.text_input("Discriminator Learning Rate", value=str(PARAMS_JSON["discriminator_compile_params"]["learning_rate"])))),
        "metrics": [
            "binary_accuracy"
        ]
    }

    return discriminator_params, discriminator_compile_params

# Main Functions
def formulagan_train_encdec():
    # Title
    st.markdown("# FormulaGAN Train: Encoder-Decoder")

    # Load Inputs
    # Load JSON Params
    PARAMS_JSON = json.load(open(PATHS["params"], "r"))
    # Main Params
    model_path = st.text_input("Model Path", PARAMS_JSON["model_path"])
    dataset_path = st.text_input("Dataset Path", PARAMS_JSON["dataset_path"])
    tokenizer_path = st.text_input("Tokenizer Path", PARAMS_JSON["tokenizer_path"])
    LOAD_TOKENIZER = True
    LOAD_MODEL = st.checkbox("Load model from path? (Must have Model_Best.h5 and TrainHistory.json)", PARAMS_JSON["LOAD_MODEL"])
    # Dataset Params
    col1, col2 = st.columns(2)
    DATASET_LOAD_COUNT =  col1.number_input("Train Dataset Load Count (-1 for whole dataset)", -1, 100000, value=PARAMS_JSON["DATASET_LOAD_COUNT"])
    VAL_DATASET_LOAD_COUNT = col2.number_input("Validation Dataset Load Count (-1 for whole dataset)", -1, 100000, value=PARAMS_JSON["VAL_DATASET_LOAD_COUNT"])
    image_size = tuple(PARAMS_JSON["image_size"])
    MAX_SEQUENCE_LENGTH = PARAMS_JSON["MAX_SEQUENCE_LENGTH"]
    col1, col2 = st.columns(2)
    col1.markdown(f"Image Size: **{image_size}**")
    col2.markdown(f"Max Sequence Length: **{MAX_SEQUENCE_LENGTH}**")
    # Train Params
    col1, col2 = st.columns(2)
    epochs = col1.number_input("Epochs", 1, 100, value=PARAMS_JSON["epochs"])
    batch_size = col2.number_input("Batch Size", 1, 100, value=PARAMS_JSON["batch_size"])
    buffer_size = 1024
    TRAIN_MODE = "simple_teacher_forcing"
    # Model Params
    # Encoder Params
    encoder_params = UI_LoadEncoderParams(PARAMS_JSON)
    # Decoder Params
    decoder_params = UI_LoadDecoderParams(PARAMS_JSON)
    # Compile Params
    compile_params = {
        "loss_fn": CategoricalCrossentropy(),
        # "loss_fn": SparseCategoricalCrossentropy(),
        "optimizer": Adam(learning_rate=float(st.text_input("Learning Rate", value=str(PARAMS_JSON["compile_params"]["learning_rate"])))),
        "metrics": [
            "categorical_accuracy", 
            # "sparse_categorical_accuracy"
        ]
    }
    datagen_params = {
        "norm": True,
        "norm_invert": False,
        "one_hot": [False, True]
    }

    # Inits
    DATASET_TRAIN = {"tokenizer": Dataset_Tokenizer_LoadTokenizer(tokenizer_path) if LOAD_TOKENIZER else None}
    if len(image_size) == 2: image_size = image_size + [int(DATASET_PARAMS["images"]["n_channels"])]
    elif len(image_size) == 3: DATASET_PARAMS["images"]["n_channels"] = image_size[2]
    DATASET_PARAMS["images"]["size"] = image_size
    DATASET_PARAMS["codes"]["max_sequence_length"] = MAX_SEQUENCE_LENGTH
    # Load Datasets
    # Load Train Dataset
    DATASET_PARAMS["images"]["size"] = image_size
    DATASET_TRAIN = Dataset_GetDataset(
        dataset_path, tokenizer=DATASET_TRAIN["tokenizer"], 
        expand_sequences=False, 
        batch_size=batch_size, buffer_size=buffer_size, 
        N=DATASET_LOAD_COUNT
    )
    st.sidebar.markdown(f"Loaded {DATASETGEN_SIZES[TRAIN_MODE](DATASET_TRAIN['train'])} train samples.")
    # Load Val Dataset
    DATASET_VAL = Dataset_GetDataset(
        dataset_path, tokenizer=DATASET_TRAIN["tokenizer"], mode="val", 
        expand_sequences=False, 
        batch_size=batch_size, buffer_size=buffer_size, 
        N=VAL_DATASET_LOAD_COUNT
    )
    st.sidebar.markdown(f"Loaded {DATASETGEN_SIZES[TRAIN_MODE](DATASET_VAL['val'])} val samples.")
    # Load/Create Model
    if not LOAD_MODEL:
        # Init Model
        # Encoder
        # print("Preparing Encoder...")
        encoder = ENCODERS[encoder_params["type"]](image_size, **encoder_params)
        # Decoder
        # print("Preparing Decoder...")
        decoder = DECODERS["token_output"][decoder_params["type"]](
            encoder["output"],
            vocab_size=DATASET_TRAIN["tokenizer"].vocabulary_size(),
            max_sequence_length=DATASET_PARAMS["codes"]["max_sequence_length"],
            **decoder_params
        )
        # Form Model
        model = Model_InitModel(encoder["input"], decoder["input"], decoder["output"])
        # Compile Model
        model = Model_Compile(model, **compile_params)
    else:
        model = Model_LoadModel(model_path)
    # Display Model
    # print("\n\n")
    # print("Model Summary:")
    # print(model.summary())
    # Plot Model
    I_model_plot = plot_model(model, to_file=os.path.join(PATHS["temp"], "Model.png"), show_shapes=True)
    st.image(os.path.join(PATHS["temp"], "Model.png"), caption="Model", use_column_width=True)

    # Process Inputs
    if st.button("Train"):
        # Load Train History
        TRAIN_HISTORY = None if not LOAD_MODEL else json.load(open(os.path.join(os.path.dirname(model_path), "TrainHistory.json"), "r"))
        # Form Inputs
        st.markdown("Model Training... (See terminal for progress)")
        print("Training Model...")
        INPUTS = DATASET_TRAIN
        INPUTS["val"] = DATASET_VAL["val"]
        datagen_func = functools.partial(
            DATASETGEN_FUNCS[TRAIN_MODE], 
            epochs=epochs, vocab_size=INPUTS["tokenizer"].vocabulary_size(), 
            **datagen_params
        )
        # Train Model
        TRAIN_HISTORY = Model_Train(
            model, INPUTS, datagen_func, 
            n_epochs=epochs,
            best_model_path=os.path.join(os.path.dirname(model_path), "Model_Best.h5"), 
            metric_name=compile_params["metrics"][0],
            TRAIN_HISTORY=TRAIN_HISTORY
        )
        # Save Final Model
        # Model_SaveModel(model, os.path.join(os.path.dirname(model_path), "Model_Final.h5"))
        # Save Train History
        json.dump(TRAIN_HISTORY, open(os.path.join(os.path.dirname(model_path), "TrainHistory.json"), "w"), indent=4)
        # Plot Train History
        plots = ["plot"]#, "scatter"]
        PLOT_FIGS = Plot_TrainHistory_EncDec(TRAIN_HISTORY, plots=plots, display=False)["figures"]
        st.markdown("## Train History")
        for k in PLOT_FIGS.keys():
            st.plotly_chart(PLOT_FIGS[k])

def formulagan_train_gan():
    # Title
    st.markdown("# FormulaGAN Train: GAN")

    # Load Inputs
    # Load JSON Params
    PARAMS_JSON = json.load(open(PATHS["params"], "r"))
    # Main Params
    model_path = st.text_input("Model Dir", PARAMS_JSON["model_path"])
    dataset_path = st.text_input("Dataset Path", PARAMS_JSON["dataset_path"])
    tokenizer_path = st.text_input("Tokenizer Path", PARAMS_JSON["tokenizer_path"])
    LOAD_TOKENIZER = True
    LOAD_MODEL = st.checkbox("Load model from path? (Must have Model_Generator_Best.h5, Model_Discriminator_Best.h5 and TrainHistory.json)", PARAMS_JSON["LOAD_MODEL"])
    # Dataset Params
    col1, col2 = st.columns(2)
    DATASET_LOAD_COUNT =  col1.number_input("Train Dataset Load Count (-1 for whole dataset)", -1, 100000, value=PARAMS_JSON["DATASET_LOAD_COUNT"])
    VAL_DATASET_LOAD_COUNT = col2.number_input("Validation Dataset Load Count (-1 for whole dataset)", -1, 100000, value=PARAMS_JSON["VAL_DATASET_LOAD_COUNT"])
    image_size = tuple(PARAMS_JSON["image_size"])
    MAX_SEQUENCE_LENGTH = PARAMS_JSON["MAX_SEQUENCE_LENGTH"]
    col1, col2 = st.columns(2)
    col1.markdown(f"Image Size: **{image_size}**")
    col2.markdown(f"Max Sequence Length: **{MAX_SEQUENCE_LENGTH}**")
    # Train Params
    col1, col2 = st.columns(2)
    epochs = col1.number_input("Epochs", 1, 100, value=PARAMS_JSON["epochs"])
    batch_size = col2.number_input("Batch Size", 1, 100, value=PARAMS_JSON["batch_size"])
    buffer_size = 1024
    TRAIN_MODE = "gan_teacher_forcing"
    # Model Params
    # Encoder Params
    encoder_params = UI_LoadEncoderParams(PARAMS_JSON)
    # Decoder Params
    decoder_params = UI_LoadDecoderParams(PARAMS_JSON)
    # Discriminator Params
    discriminator_params, discriminator_compile_params = UI_LoadDiscriminatorParams(PARAMS_JSON, encoder_params, decoder_params, MAX_SEQUENCE_LENGTH)
    # Compile Params
    compile_params = {
        "loss_fn": CategoricalCrossentropy(),
        # "loss_fn": SparseCategoricalCrossentropy(),
        "optimizer": Adam(learning_rate=float(st.text_input("Generator Learning Rate", value=str(PARAMS_JSON["compile_params"]["learning_rate"])))),
        "metrics": [
            "categorical_accuracy", 
            # "sparse_categorical_accuracy"
        ]
    }
    datagen_params = {
        "norm": True,
        "norm_invert": False,
        "one_hot": [False, True, True]
    }

    # Inits
    DATASET_TRAIN = {"tokenizer": Dataset_Tokenizer_LoadTokenizer(tokenizer_path) if LOAD_TOKENIZER else None}
    if len(image_size) == 2: image_size = image_size + [int(DATASET_PARAMS["images"]["n_channels"])]
    elif len(image_size) == 3: DATASET_PARAMS["images"]["n_channels"] = image_size[2]
    DATASET_PARAMS["images"]["size"] = image_size
    DATASET_PARAMS["codes"]["max_sequence_length"] = MAX_SEQUENCE_LENGTH
    # Load Datasets
    # Load Train Dataset
    DATASET_PARAMS["images"]["size"] = image_size
    DATASET_TRAIN = Dataset_GetDataset(
        dataset_path, tokenizer=DATASET_TRAIN["tokenizer"], 
        expand_sequences=False, 
        batch_size=batch_size, buffer_size=buffer_size, 
        N=DATASET_LOAD_COUNT
    )
    st.sidebar.markdown(f"Loaded {DATASETGEN_SIZES[TRAIN_MODE](DATASET_TRAIN['train'])} train samples.")
    # Load Val Dataset
    DATASET_VAL = Dataset_GetDataset(
        dataset_path, tokenizer=DATASET_TRAIN["tokenizer"], mode="val", 
        expand_sequences=False, 
        batch_size=batch_size, buffer_size=buffer_size, 
        N=VAL_DATASET_LOAD_COUNT
    )
    st.sidebar.markdown(f"Loaded {DATASETGEN_SIZES[TRAIN_MODE](DATASET_VAL['val'])} val samples.")
    # Load/Create Model
    if not LOAD_MODEL:
        # Create Generator
        # Encoder
        # print("Preparing Encoder...")
        encoder = ENCODERS[encoder_params["type"]](image_size, **encoder_params)
        # Decoder
        # print("Preparing Decoder...")
        decoder = DECODERS["token_output"][decoder_params["type"]](
            encoder["output"],
            vocab_size=DATASET_TRAIN["tokenizer"].vocabulary_size(),
            max_sequence_length=DATASET_PARAMS["codes"]["max_sequence_length"],
            **decoder_params
        )
        # Form Generator
        generator = Model_InitModel(encoder["input"], decoder["input"], decoder["output"])
        # Compile Generator
        generator = Model_Compile(generator, **compile_params)

        # Create Discriminator
        # print("Preparing Discriminator...")
        discriminator = DISCRIMINATORS["token"](vocab_size=DATASET_TRAIN["tokenizer"].vocabulary_size(), **discriminator_params)
        # Compile Discriminator
        # discriminator = Model_Compile(discriminator, **discriminator_compile_params)
    else:
        generator = Model_LoadModel(os.path.join(model_path, "Model_Generator_Best.h5"))
        discriminator = Model_LoadModel(os.path.join(model_path, "Model_Discriminator_Best.h5"))
    # Create GAN Generator
    # print("Preparing GAN Generator...")
    gan_gen = GANModel_GANGenerator(generator, discriminator, decoder_concat_name="decoder_concat")
    # Compile GAN Generator
    gan_gen = Model_Compile(gan_gen, **discriminator_compile_params)
    # Create GAN Discriminator
    # print("Preparing GAN Discriminator...")
    gan_disc = GANModel_GANDiscriminator(generator, discriminator, decoder_concat_name="decoder_concat")
    # Compile GAN Discriminator
    gan_disc = Model_Compile(gan_disc, **discriminator_compile_params)
    # Plot Model
    I_gen_plot = plot_model(generator, to_file=os.path.join(PATHS["temp"], "Model_Generator.png"), show_shapes=True)
    st.image(os.path.join(PATHS["temp"], "Model_Generator.png"), caption="Generator", use_column_width=True)
    I_disc_plot = plot_model(discriminator, to_file=os.path.join(PATHS["temp"], "Model_Discriminator.png"), show_shapes=True)
    st.image(os.path.join(PATHS["temp"], "Model_Discriminator.png"), caption="Discriminator", use_column_width=True)

    # Process Inputs
    if st.button("Train"):
        # Load Train History
        TRAIN_HISTORY = None if not LOAD_MODEL else json.load(open(os.path.join(model_path, "TrainHistory.json"), "r"))
        # Form Inputs
        st.markdown("Model Training... (See terminal for progress)")
        print("Training Model...")
        INPUTS = DATASET_TRAIN
        INPUTS["val"] = DATASET_VAL["val"]
        datagen_funcs = {
            "train": functools.partial(
                DATASETGEN_FUNCS[TRAIN_MODE], 
                epochs=epochs, vocab_size=INPUTS["tokenizer"].vocabulary_size(), 
                **datagen_params
            ),
            "val": {
                "generator": functools.partial(
                    DATASETGEN_FUNCS["simple_teacher_forcing"], 
                    epochs=epochs, vocab_size=INPUTS["tokenizer"].vocabulary_size(), 
                    **datagen_params
                ),
                "discriminator": functools.partial(
                    DATASETGEN_FUNCS["gan_teacher_forcing_discriminator"], 
                    epochs=epochs, vocab_size=INPUTS["tokenizer"].vocabulary_size(), 
                    **datagen_params
                )
            }
        }
        # Train Model
        TRAIN_HISTORY = GANModel_Train(
            generator, discriminator, gan_gen, gan_disc, 
            INPUTS, datagen_funcs, 
            n_epochs=epochs,
            best_model_path=model_path, 
            SEPARATE_DISC_LOSS=False, SPLIT_DISC_BATCH=True, SHUFFLE_BATCH=False, SHUFFLE_DISC_SAMPLES=True,
            ROUND_OFF_DIGITS=8, RECORD_INTERVAL=50, 
            TRAIN_HISTORY=TRAIN_HISTORY
        )
        # Save Final Model
        # Model_SaveModel(generator, os.path.join(model_path, "Model_Generator_Final.h5"))
        # Model_SaveModel(discriminator, os.path.join(model_path, "Model_Discriminator_Final.h5"))
        # Save Train History
        json.dump(TRAIN_HISTORY, open(os.path.join(model_path, "TrainHistory.json"), "w"), indent=4)
        # Plot Train History
        plots = ["plot"]#, "scatter"]
        PLOT_FIGS = Plot_TrainHistory_GAN(TRAIN_HISTORY, plots=plots, display=False)["figures"]
        st.markdown("## Train History")
        for k in PLOT_FIGS.keys():
            st.plotly_chart(PLOT_FIGS[k])

# Mode Vars
MODES = {
    "FormulaGAN Train: Encoder-Decoder": formulagan_train_encdec,
    "FormulaGAN Train: GAN": formulagan_train_gan,
}

# App Functions
def app_main():
    # Title
    st.markdown("# Formula GAN Project - Train")
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

# Good Examples for IM2LATEX Dataset:
# 7785, 