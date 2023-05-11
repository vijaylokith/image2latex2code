"""
Model
"""

# Imports
from .Dataset import *
from .Evaluate import *
from .ModelLibrary.Model_EncDec import *
from .ModelLibrary.Model_GAN import *

# Main Functions
# Load / Save Functions
def Model_LoadModel(path):
    '''
    Model - Load Model
    '''
    return load_model(path)

def Model_SaveModel(model, path):
    '''
    Model - Save Model
    '''
    return model.save(path)

# Compile Model Function
def Model_Compile(model, loss_fn=None, optimizer=None, metrics=["sparse_categorical_accuracy"], **params):
    '''
    Model - Compile Model
    '''
    if loss_fn is None: loss_fn = SparseCategoricalCrossentropy(from_logits=False, reduction="none")
    if optimizer is None: optimizer = Adam()

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    return model

# Model Predict Functions
def Model_Predict(
    model, tokenizer, lookups, 
    I_path=None, I=None, 
    DECODER_INPUT_TEXT="<start>", 
    **params
    ):
    '''
    Model - Predict Sequence for an Image
    '''
    if I is None: I = Dataset_LoadImage_Tensorflow(I_path)
    C = np.array(tokenizer(DECODER_INPUT_TEXT)).reshape((1, -1))

    # Predict
    start_length = len(DECODER_INPUT_TEXT.split(" "))
    max_length = C.shape[-1]
    enc_inp = I.numpy().reshape((1, I.shape[0], I.shape[1]))
    for i in range(start_length-1, max_length-1):
        token_pred_onehot = model.predict([enc_inp, C])
        token_pred = np.argmax(token_pred_onehot, axis=-1)
        # Check end token
        if token_pred == lookups["word_to_index"]("<end>"):
            break
        # Set Pred Token
        C[:, i+1] = token_pred
    C[:, i+1] = lookups["word_to_index"]("<end>")

    # Decode
    seq_pred = []
    for token in C[0, :i+2]:
        word_pred = lookups["index_to_word"](token).numpy().decode("utf-8")
        seq_pred.append(word_pred)
    seq_pred = " ".join(seq_pred)

    return seq_pred

# Model Evaluation Functions
def ModelEval_SequencePrediction_VariedInputSeq(
    model, code,
    tokenizer, lookups,
    I_path=None, I=None, 
    **params
    ):
    '''
    Model - Evaluate Model - Sequence Prediction
    VariedInputSeq
        - Predict sequence with varied input sequence
        - start from "<start>" and try with new input sequences by adding next token
        - Check accuracy of prediction with more and more tokens given in input sequence
    '''
    # Init
    print("Sequence Prediction - Varied Input Seq")
    code_tokens = code.split(" ")
    # Predict
    print("Predicted Sequences:")
    seqs_pred = []
    tokens_inp = []
    for i in tqdm(range(len(code_tokens)-1)):
        # Predict
        next_token = code_tokens[i]
        tokens_inp.append(next_token)
        seq_inp = " ".join(tokens_inp)
        seq_pred = Model_Predict(model, tokenizer, lookups, I_path=I_path, I=I, DECODER_INPUT_TEXT=seq_inp)
        seqs_pred.append(seq_pred)
        # Analyse
        true_tokens = np.array(code_tokens)
        pred_tokens = np.array(seq_pred.split(" "))
        next_token_match = code_tokens[i+1] == pred_tokens[i+1]
        min_length = min(true_tokens.shape[0], pred_tokens.shape[0])
        n_matches = sum(true_tokens[:min_length] == pred_tokens[:min_length])
        # Display
        print("Seq Lengths: (Input, True, Pred)", i+1, true_tokens.shape[0], pred_tokens.shape[0])
        print("Next Token Match:", next_token_match)
        print("Matches:", n_matches, "/", min_length)
        print("Input Sequence    :", " ".join(tokens_inp[:i+1]))
        print("True Sequence     :", code)
        print("Predicted Sequence:", seqs_pred[i])
        print()

def ModelEval_Scores(
    model, inputs, 
    scores=list(SCORE_FUNCS.keys())
    ):
    '''
    Model - Evaluate Model - Scores
    '''
    # Get Data
    TOKENIZER = inputs["tokenizer"]
    LOOKUPS = inputs["lookups"]
    DATASET = inputs["test"]["dataset"]
    NUM_STEPS = inputs["test"]["n_steps"]
    datagen_func = functools.partial(
        DATASETGEN_FUNCS["simple"], 
        epochs=1, norm=True, norm_invert=False, one_hot=[False, False]
    )
    DATASET_ITERATOR = datagen_func(DATASET["images"], DATASET["codes"])

    # Init
    Scores = {s: 0.0 for s in scores}
    Sequences = {"true": [], "pred": []}
    batch_pbar = tqdm(range(NUM_STEPS))
    batch_pbar.set_description(f"Score Eval")

    # Predict
    for batch_i in batch_pbar:
        # Get
        DI, DO = next(DATASET_ITERATOR)
        enc_inp, dec_inp = DI
        dec_out = DO
        # Predict
        C = np.tile(np.array(TOKENIZER("<start>")), (enc_inp.shape[0], 1))
        for i in range(C.shape[1]-1):
            token_pred_onehot = model.predict([enc_inp, C])
            token_pred = np.argmax(token_pred_onehot, axis=-1)
            # Set Pred Token
            C[:, i+1] = token_pred
        C[:, i+1] = LOOKUPS["word_to_index"]("<end>")
        # Decode
        for i in range(enc_inp.shape[0]):
            seq_pred = []
            seq_true = []
            for token in C[i]:
                word_pred = LOOKUPS["index_to_word"](token).numpy().decode("utf-8")
                seq_pred.append(word_pred)
                if word_pred == "<end>": break
            for token in dec_out[i]:
                word_true = LOOKUPS["index_to_word"](token).numpy().decode("utf-8")
                seq_true.append(word_true)
                if word_true == "<end>": break
            Sequences["pred"].append(seq_pred)
            Sequences["true"].append(seq_true)

    # Evaluate
    for s in scores:
        Scores[s] = SCORE_FUNCS[s](Sequences["true"], Sequences["pred"])

    # Display
    print("Scores:")
    for s in scores:
        print(f"{s}: {Scores[s]}")
    print()

    return Scores, Sequences

# RunCode
print("Reloaded Model Module.")