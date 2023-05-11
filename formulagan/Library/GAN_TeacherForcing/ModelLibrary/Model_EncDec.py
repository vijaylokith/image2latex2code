"""
Encoder-Decoder Model
"""

# Imports
from .Encoder import *
from .Decoder import *

# Main Functions
# Encoder Decoder Functions
def Model_InitModel(encoder_input, decoder_input, decoder_output):
    '''
    Model - Initialize Model
    '''
    # Form Model
    model = Model([encoder_input, decoder_input], decoder_output)

    return model

# Train Functions
def Model_Train(
    model, inputs, datagen_func, 
    n_epochs=1, 
    best_model_path="Models/best_model/Model_Best.h5", metric_name="sparse_categorical_accuracy",
    TRAIN_HISTORY=None,
    **params
    ):
    '''
    Model - Train Model
    '''
    # Get Data
    DATASET = inputs["train"]["dataset"]
    NUM_STEPS = inputs["train"]["n_steps"]
    
    DATASET_VAL = inputs["val"]["dataset"]
    NUM_STEPS_VAL = inputs["val"]["n_steps"]

    callbacks = []
    # Enable Model Checkpointing
    ModelCheckpointFunc = ModelCheckpoint(
        best_model_path,
        monitor="val_" + metric_name,
        verbose=1,
        save_best_only=True,
        mode="max",
        save_freq="epoch"
    )
    callbacks.append(ModelCheckpointFunc)

    # Train Model
    train_out = model.fit(
        datagen_func(DATASET["images"], DATASET["codes"]), 
        steps_per_epoch=NUM_STEPS,

        validation_data=datagen_func(DATASET_VAL["images"], DATASET_VAL["codes"]), 
        validation_steps=NUM_STEPS_VAL,

        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Record History
    if TRAIN_HISTORY is None:
        TRAIN_HISTORY = {
            "epoch": n_epochs,
            "train": {
                "loss": train_out.history["loss"],
                "metric": train_out.history[metric_name],
            },
            "val": {
                "loss": train_out.history["val_loss"],
                "metric": train_out.history["val_" + metric_name],
            }
        }
    else:
        TRAIN_HISTORY["epoch"] += n_epochs
        TRAIN_HISTORY["train"]["loss"].extend(train_out.history["loss"])
        TRAIN_HISTORY["train"]["metric"].extend(train_out.history[metric_name])
        TRAIN_HISTORY["val"]["loss"].extend(train_out.history["val_loss"])
        TRAIN_HISTORY["val"]["metric"].extend(train_out.history["val_" + metric_name])
    return TRAIN_HISTORY

# Plot Functions
def Plot_ListData_Func(n, data, plots=["plot", "scatter"], title="Data", display=True):
    '''
    Plot - Plot List Data
    '''
    FIG = plt.figure()
    if "plot" in plots: plt.plot(range(n), data)
    if "scatter" in plots: plt.scatter(range(n), data)
    plt.title(title)
    plt.legend()
    if display: plt.show()

    return FIG

def Plot_TrainHistory_EncDec(
    TRAIN_HISTORY, 
    plots=["plot", "scatter"],
    display=True
    ):
    '''
    Plot - Plot Encoder-Decoder Training History Losses and Metrics
    '''
    # Init
    plotData = {
        "n": len(TRAIN_HISTORY["train"]["loss"]),
        "val_n": len(TRAIN_HISTORY["val"]["loss"]),
        "loss": TRAIN_HISTORY["train"]["loss"],
        "metric": TRAIN_HISTORY["train"]["metric"],
        "val_loss": TRAIN_HISTORY["val"]["loss"],
        "val_metric": TRAIN_HISTORY["val"]["metric"]
    }

    # Loss Plot
    FIG_LOSS = Plot_ListData_Func(plotData["n"], plotData["loss"], plots=plots, title="Loss", display=display)
    # Metric Plot
    FIG_METRIC = Plot_ListData_Func(plotData["n"], plotData["metric"], plots=plots, title="Metric", display=display)
    # Val Loss Plot
    FIG_VALLOSS = Plot_ListData_Func(plotData["val_n"], plotData["val_loss"], plots=plots, title="Val Loss", display=display)
    # Val Metric Plot
    FIG_VALMETRIC = Plot_ListData_Func(plotData["val_n"], plotData["val_metric"], plots=plots, title="Val Metric", display=display)

    PLOT_OUT = {
        "figures": {
            "loss": FIG_LOSS,
            "metric": FIG_METRIC,
            "val_loss": FIG_VALLOSS,
            "val_metric": FIG_VALMETRIC
        }
    }
    return PLOT_OUT