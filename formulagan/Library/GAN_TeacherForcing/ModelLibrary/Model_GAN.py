"""
GAN Model
"""

# Imports
from .Encoder import *
from .Decoder import *
from .Discriminator import *

# Main Functions
# GAN Functions
def GANModel_TrueDataRepresentationModel(
    generator, 
    decoder_concat_name="decoder_concat",
    **params
    ):
    '''
    GANModel - True Data Representation Model
    Inputs:
        - Image (BatchSize, Height, Width)
        - Input Sequence (BatchSize, SequenceLength, VocabSize)
    Outputs:
        - True Data Representation (BatchSize, SequenceLength, Img_EmbeddingDim + Code_EmbeddingDim)
    '''
    # Get Encoder and Decoder Inputs from Generator
    gen_enc_inp, gen_dec_inp = generator.input
    # Get True Data Representation from Decoder Concat Layer
    gen_truedatarep_out = generator.get_layer(decoder_concat_name).output
    # Define True Data Representation Model
    model = Model([gen_enc_inp, gen_dec_inp], gen_truedatarep_out)

    return model

def GANModel_GANGenerator(
    generator, discriminator,
    decoder_concat_name="decoder_concat",
    **params
    ):
    '''
    GANModel - Generator GAN Model - Used to train Generator part of GAN
    Inputs:
        - Image (BatchSize, Height, Width)
        - Input Sequence (BatchSize, SequenceLength, VocabSize)
    Outputs:
        - Discriminator Output (BatchSize, 1)
    '''
    # Get Encoder and Decoder Inputs from Generator
    gen_enc_inp, gen_dec_inp = generator.input
    # Get True Data Representation from Decoder Concat Layer
    gen_truedatarep_out = generator.get_layer(decoder_concat_name).output
    # Get Predicted Token Output from the Generator
    gen_predtoken_out = generator.output
    # Connect Predicted Token Output and Label Input from Generator as Inputs to Discriminator
    gan_output = discriminator([gen_truedatarep_out, gen_predtoken_out])
    # Define GAN Model
    model = Model([gen_enc_inp, gen_dec_inp], gan_output)

    return model

def GANModel_GANDiscriminator(
    generator, discriminator,
    decoder_concat_name="decoder_concat",
    **params
    ):
    '''
    GANModel - Discriminator GAN Model - Used to train Discriminator part of GAN
    Inputs:
        - Image (BatchSize, Height, Width)
        - Input Sequence (BatchSize, SequenceLength, VocabSize)
        - Output Token/Sequence (BatchSize, SequenceLength, VocabSize) or (BatchSize, VocabSize) depending on Decoder
    Outputs:
        - Discriminator Output (BatchSize, 1)
    '''
    # Get Encoder and Decoder Inputs from Generator
    gen_enc_inp, gen_dec_inp = generator.input
    # Get True Data Representation from Decoder Concat Layer
    gen_truedatarep_out = generator.get_layer(decoder_concat_name).output
    # Assign Input to Token Output
    token_out = Input(shape=generator.output.shape[1:])
    # Connect Predicted Token Output and Label Input from Generator as Inputs to Discriminator
    gan_output = discriminator([gen_truedatarep_out, token_out])
    # Define GAN Model
    model = Model([gen_enc_inp, gen_dec_inp, token_out], gan_output)

    return model

# Train Functions
def GANModel_Train(
    g_model, d_model, gan_gen_model, gan_disc_model,
    inputs, datagen_funcs, 
    n_epochs=1, 
    best_model_path="Models/best_model/", 

    SEPARATE_DISC_LOSS=True, SPLIT_DISC_BATCH=False, SHUFFLE_BATCH=False, SHUFFLE_DISC_SAMPLES=False,
    ROUND_OFF_DIGITS=4, RECORD_INTERVAL=1,
    TRAIN_HISTORY=None,

    **params):
    '''
    GANModel - Train GAN Model
    '''
    # Get Data
    DATASET = inputs["train"]["dataset"]
    NUM_STEPS = inputs["train"]["n_steps"]
    DATASET_VAL = inputs["val"]["dataset"]
    NUM_STEPS_VAL = inputs["val"]["n_steps"]

    DATASET_ITERATOR = datagen_funcs["train"](DATASET["images"], DATASET["codes"])
    VAL_DATASET_ITERATOR = datagen_funcs["val"]["generator"](DATASET_VAL["images"], DATASET_VAL["codes"])
    VAL_DISC_DATASET_ITERATOR = datagen_funcs["val"]["discriminator"](DATASET_VAL["images"], DATASET_VAL["codes"])

    # History
    if TRAIN_HISTORY is None:
        epoch_data = {
            "cur_epoch": 0,
            "overall": [],
            "best": {
                "epoch": None,
                "loss": {
                    "generator": None,
                    "discriminator": None
                },
                "metric": {
                    "generator": None,
                    "discriminator": None
                }
            }
        }
        val_data = {
            "overall": [],
            "best": {
                "epoch": None,
                "loss": {
                    "generator": None,
                    "discriminator": None
                },
                "metric": {
                    "generator": None,
                    "discriminator": None
                }
            }
        }
    else:
        epoch_data = TRAIN_HISTORY["epoch_data"]
        val_data = TRAIN_HISTORY["val_data"]
    
    # Train Epoch Loop
    for epoch_i in range(epoch_data["cur_epoch"], epoch_data['cur_epoch']+n_epochs):
        # Init Progress Bar
        batch_pbar = tqdm(range(NUM_STEPS))
        batch_pbar.set_description(f"Epoch {epoch_i+1}/{epoch_data['cur_epoch']+n_epochs}")
        # Train Batch Loop
        epoch_data["current"] = {
            "loss": {
                "generator": None,
                "discriminator": None
            },
            "metric": {
                "generator": None,
                "discriminator": None
            }
        }
        for batch_i in batch_pbar:
            # Get Data
            DI, DO, YT = next(DATASET_ITERATOR)
            # enc_inp: (ExpandedSize, Height, Width)
            # dec_inp: (ExpandedSize, SequenceLength, VocabSize)
            # dec_out: (ExpandedSize, VocabSize)
            # y_true: (ExpandedSize, SequenceLength, VocabSize) or (ExpandedSize, VocabSize) depending on type of decoder
            enc_inp, dec_inp = DI
            dec_out = DO
            y_true = YT
            # Shuffle Samples
            if SHUFFLE_BATCH:
                shuffle_order = tf.random.shuffle(tf.range(enc_inp.shape[0]))
                enc_inp = tf.gather(enc_inp, shuffle_order)
                dec_inp = tf.gather(dec_inp, shuffle_order)
                dec_out = tf.gather(dec_out, shuffle_order)

            # Discriminator Training
            # Set Discriminator part to trainable and Generator part to not trainable
            d_model.trainable = True
            g_model.trainable = False
            # Get Real "Valid" Samples
            REAL_enc_inp = enc_inp[::2] if SPLIT_DISC_BATCH else enc_inp
            REAL_dec_inp = dec_inp[::2] if SPLIT_DISC_BATCH else dec_inp
            REAL_dec_out = dec_out[::2] if SPLIT_DISC_BATCH else dec_out
            REAL_gan_out = tf.ones((REAL_enc_inp.shape[0], 1), dtype=tf.float32)
            # Generate Fake "Invalid" Samples using Generator
            FAKE_enc_inp = enc_inp[1::2] if SPLIT_DISC_BATCH else enc_inp
            FAKE_dec_inp = dec_inp[1::2] if SPLIT_DISC_BATCH else dec_inp
            FAKE_dec_out = g_model.predict([FAKE_enc_inp, FAKE_dec_inp])
            FAKE_gan_out = tf.zeros((FAKE_enc_inp.shape[0], 1), dtype=tf.float32)
            
            if SEPARATE_DISC_LOSS:
                # Train Discriminator on Real and Fake Samples
                d_out_real = gan_disc_model.train_on_batch([REAL_enc_inp, REAL_dec_inp, REAL_dec_out], REAL_gan_out)
                d_out_fake = gan_disc_model.train_on_batch([FAKE_enc_inp, FAKE_dec_inp, FAKE_dec_out], FAKE_gan_out)
                d_out = {
                    "loss": {
                        "real": round(d_out_real[0], ROUND_OFF_DIGITS),
                        "fake": round(d_out_fake[0], ROUND_OFF_DIGITS)
                    },
                    "metric": {
                        "real": round(d_out_real[1], ROUND_OFF_DIGITS),
                        "fake": round(d_out_fake[1], ROUND_OFF_DIGITS)
                    }
                }
            else:
                # Concat Real and Fake Samples
                DISC_enc_inp = tf.concat([REAL_enc_inp, FAKE_enc_inp], axis=0)
                DISC_dec_inp = tf.concat([REAL_dec_inp, FAKE_dec_inp], axis=0)
                DISC_dec_out = tf.concat([REAL_dec_out, FAKE_dec_out], axis=0)
                DISC_gan_out = tf.concat([REAL_gan_out, FAKE_gan_out], axis=0)
                # Shuffle Samples
                if SHUFFLE_DISC_SAMPLES:
                    shuffle_order = tf.random.shuffle(tf.range(DISC_enc_inp.shape[0]))
                    DISC_enc_inp = tf.gather(DISC_enc_inp, shuffle_order)
                    DISC_dec_inp = tf.gather(DISC_dec_inp, shuffle_order)
                    DISC_dec_out = tf.gather(DISC_dec_out, shuffle_order)
                    DISC_gan_out = tf.gather(DISC_gan_out, shuffle_order)
                # Train Discriminator on Concatenated Samples
                d_out_gan = gan_disc_model.train_on_batch([DISC_enc_inp, DISC_dec_inp, DISC_dec_out], DISC_gan_out)
                d_out = {
                    "loss": {
                        "full": round(d_out_gan[0], ROUND_OFF_DIGITS)
                    },
                    "metric": {
                        "full": round(d_out_gan[1], ROUND_OFF_DIGITS)
                    }
                }
                # CleanUp
                del DISC_enc_inp, DISC_dec_inp, DISC_dec_out, DISC_gan_out
            # CleanUp
            del REAL_gan_out, FAKE_dec_out, FAKE_gan_out

            # Generator Training
            # Set Generator part to trainable and Discriminator part to not trainable
            g_model.trainable = True
            d_model.trainable = False
            # Get Predicted Samples from Generator
            GEN_enc_inp = enc_inp
            GEN_dec_inp = dec_inp
            GEN_dec_out = dec_out
            GEN_gan_out = tf.ones((GEN_enc_inp.shape[0], 1), dtype=tf.float32)
            # Train Generator using Generator's Error
            g_out_gen = g_model.train_on_batch([GEN_enc_inp, GEN_dec_inp], GEN_dec_out)
            # Train Generator using Discriminator's Error
            g_out_gan = gan_gen_model.train_on_batch([GEN_enc_inp, GEN_dec_inp], GEN_gan_out)
            g_out = {
                "loss": {
                    "gan": round(g_out_gan[0], ROUND_OFF_DIGITS),
                    "gen": round(g_out_gen[0], ROUND_OFF_DIGITS)
                },
                "metric": {
                    "gan": round(g_out_gan[1], ROUND_OFF_DIGITS),
                    "gen": round(g_out_gen[1], ROUND_OFF_DIGITS)
                }
            }
            # CleanUp
            del GEN_gan_out

            # Record
            epoch_data["current"] = {
                "loss": {
                    "generator": g_out["loss"],
                    "discriminator": d_out["loss"]
                },
                "metric": {
                    "generator": g_out["metric"],
                    "discriminator": d_out["metric"]
                }
            }
            if batch_i % RECORD_INTERVAL == 0:
                epoch_data["overall"].append(epoch_data["current"])

            # Update Progress Bar
            batch_pbar.set_postfix(epoch_data["current"])

        # Evaluate Generator and Discriminator
        gen_eval_out = g_model.evaluate(
            VAL_DATASET_ITERATOR, 
            steps=NUM_STEPS_VAL
        )
        disc_eval_out = gan_gen_model.evaluate(
            VAL_DISC_DATASET_ITERATOR, 
            steps=NUM_STEPS_VAL
        )
        val_data["overall"].append({
            "loss": {
                "generator": round(gen_eval_out[0], ROUND_OFF_DIGITS),
                "discriminator": round(disc_eval_out[0], ROUND_OFF_DIGITS)
            },
            "metric": {
                "generator": round(gen_eval_out[1], ROUND_OFF_DIGITS),
                "discriminator": round(disc_eval_out[1], ROUND_OFF_DIGITS)
            }
        })

        # Display Progress
        print(f"Epoch: {epoch_i+1}")
        print(f"Generator Loss:           {epoch_data['current']['loss']['generator']}")
        print(f"Generator Metric:         {epoch_data['current']['metric']['generator']}")
        print(f"Discriminator Loss:       {epoch_data['current']['loss']['discriminator']}")
        print(f"Discriminator Metric:     {epoch_data['current']['metric']['discriminator']}")
        print(f"Generator Val Loss:       {val_data['overall'][-1]['loss']['generator']}")
        print(f"Generator Val Metric:     {val_data['overall'][-1]['metric']['generator']}")
        print(f"Discriminator Val Loss:   {val_data['overall'][-1]['loss']['discriminator']}")
        print(f"Discriminator Val Metric: {val_data['overall'][-1]['metric']['discriminator']}")

        # Update Best
        if val_data["best"]["loss"]["generator"] is None \
            or val_data["best"]["metric"]["generator"] < val_data["overall"][-1]["metric"]["generator"]:
            # Display
            print(f"Best Epoch: {val_data['best']['epoch']} -> {epoch_i+1}")
            print(f"Best Generator Loss: {val_data['best']['loss']['generator']} -> {val_data['overall'][-1]['loss']['generator']}")
            print(f"Best Generator Metric: {val_data['best']['metric']['generator']} -> {val_data['overall'][-1]['metric']['generator']}")
            print(f"Best Discriminator Loss: {val_data['best']['loss']['discriminator']} -> {val_data['overall'][-1]['loss']['discriminator']}")
            print(f"Best Discriminator Metric: {val_data['best']['metric']['discriminator']} -> {val_data['overall'][-1]['metric']['discriminator']}")
            # Update
            epoch_data["best"]["epoch"] = epoch_i+1
            epoch_data["best"]["loss"] = epoch_data["current"]["loss"]
            epoch_data["best"]["metric"] = epoch_data["current"]["metric"]
            val_data["best"]["epoch"] = epoch_i+1
            val_data["best"]["loss"] = val_data["overall"][-1]["loss"]
            val_data["best"]["metric"] = val_data["overall"][-1]["metric"]
            # Save Models
            g_model.save(os.path.join(best_model_path, "Model_Generator_Best.h5"))
            d_model.save(os.path.join(best_model_path, "Model_Discriminator_Best.h5"))
            print("Saved new best models.")
        else:
            # Display
            print(f"Best Epoch: {val_data['best']['epoch']}")
            print(f"Generator Loss: Best = {val_data['best']['loss']['generator']}, Current = {val_data['overall'][-1]['loss']['generator']}")
            print(f"Generator Metric: Best = {val_data['best']['metric']['generator']}, Current = {val_data['overall'][-1]['metric']['generator']}")
            print(f"Discriminator Loss: Best = {val_data['best']['loss']['discriminator']}, Current = {val_data['overall'][-1]['loss']['discriminator']}")
            print(f"Discriminator Metric: Best = {val_data['best']['metric']['discriminator']}, Current = {val_data['overall'][-1]['metric']['discriminator']}")
        print()

    # Update Epochs Run
    epoch_data["cur_epoch"] += n_epochs

    TRAIN_HISTORY = {
        "epoch_data": epoch_data,
        "val_data": val_data
    }
    return TRAIN_HISTORY

# Plot Functions
def Plot_EpochData_Func(n, data, plots=["plot", "scatter"], title="Epoch Data", display=True):
    '''
    Plot - Plot Epoch Data
    '''
    FIG = plt.figure()
    if "plot" in plots:
        for k in data["generator"].keys():
            plt.plot(range(n), data["generator"][k], label=f"Gen_{k}")
        for k in data["discriminator"].keys():
            plt.plot(range(n), data["discriminator"][k], label=f"Disc_{k}")
    if "scatter" in plots:
        for k in data["generator"].keys():
            plt.scatter(range(n), data["generator"][k], label=f"Gen_{k}")
        for k in data["discriminator"].keys():
            plt.scatter(range(n), data["discriminator"][k], label=f"Disc_{k}")
    plt.title(title)
    plt.legend()
    if display: plt.show()

    return FIG

def Plot_ValData_Func(n, data, plots=["plot", "scatter"], title="Val Data", display=True):
    '''
    Plot - Plot Validation Data
    '''
    FIG = plt.figure()
    # Generator
    plt.subplot(1, 2, 1)
    if "plot" in plots:
        plt.plot(range(n), data["generator"], label="Gen")
    if "scatter" in plots:
        plt.scatter(range(n), data["generator"], label="Gen")
    plt.title(title + " - Gen")
    plt.legend()
    # Discriminator
    plt.subplot(1, 2, 2)
    if "plot" in plots:
        plt.plot(range(n), data["discriminator"], label="Disc")
    if "scatter" in plots:
        plt.scatter(range(n), data["discriminator"], label="Disc")
    plt.title(title + " - Disc")
    plt.legend()
    if display: plt.show()

    return FIG

def Plot_TrainHistory_GAN(
    TRAIN_HISTORY, 
    plots=["plot", "scatter"],
    display=True
    ):
    '''
    Plot - Plot GAN Training History Losses and Metrics
    '''
    # Init
    epoch_data = TRAIN_HISTORY["epoch_data"]
    val_data = TRAIN_HISTORY["val_data"]
    plotData = {
        "n": len(epoch_data["overall"]),
        "val_n": len(val_data["overall"]),
        "loss": {
            "generator": {k: [d["loss"]["generator"][k] for d in epoch_data["overall"]] for k in epoch_data["overall"][0]["loss"]["generator"].keys()}, 
            "discriminator": {k: [d["loss"]["discriminator"][k] for d in epoch_data["overall"]] for k in epoch_data["overall"][0]["loss"]["discriminator"].keys()}
        },
        "metric": {
            "generator": {k: [d["metric"]["generator"][k] for d in epoch_data["overall"]] for k in epoch_data["overall"][0]["metric"]["generator"].keys()}, 
            "discriminator": {k: [d["metric"]["discriminator"][k] for d in epoch_data["overall"]] for k in epoch_data["overall"][0]["metric"]["discriminator"].keys()}
        },
        "val_loss": {
            "generator": [d["loss"]["generator"] for d in val_data["overall"]],
            "discriminator": [d["loss"]["discriminator"] for d in val_data["overall"]]
        },
        "val_metric": {
            "generator": [d["metric"]["generator"] for d in val_data["overall"]],
            "discriminator": [d["metric"]["discriminator"] for d in val_data["overall"]]
        },
    }

    # Loss Plot
    FIG_LOSS = Plot_EpochData_Func(plotData["n"], plotData["loss"], plots=plots, title="Loss", display=display)
    # Metric Plot
    FIG_METRIC = Plot_EpochData_Func(plotData["n"], plotData["metric"], plots=plots, title="Metric", display=display)
    # Val Loss Plot
    FIG_VALLOSS = Plot_ValData_Func(plotData["val_n"], plotData["val_loss"], plots=plots, title="Val Loss", display=display)
    # Val Metric Plot
    FIG_VALMETRIC = Plot_ValData_Func(plotData["val_n"], plotData["val_metric"], plots=plots, title="Val Metric", display=display)

    PLOT_OUT = {
        "figures": {
            "loss": FIG_LOSS,
            "metric": FIG_METRIC,
            "val_loss": FIG_VALLOSS,
            "val_metric": FIG_VALMETRIC
        }
    }
    return PLOT_OUT