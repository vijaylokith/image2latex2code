"""
Discriminator
"""

# Imports
from .Blocks.TokenRepBlocks import *
from .Blocks.TokenCombineBlocks import *

# Main Vars
PRINT = PRINT_FUNCS["discriminator"]

# Main Functions
def Discriminator_Sequence(
    dense_n_units=[],
    dense_activation="relu",
    dense_dropout=0.0, 
    **params
    ):
    '''
    Discriminator - Sequence Type
    Inputs:
        - True Data Representation from Decoder Concat (BatchSize, SequenceLength, Img_EmbeddingDim + Code_EmbeddingDim)
        - Predicted Sequence Representation (BatchSize, SequenceLength, Code_EmbeddingDim)
    Outputs:
        - Discriminator Output (BatchSize, 1)
    '''
    # True Data Representation Input (BatchSize, SequenceLength, Img_EmbeddingDim + Code_EmbeddingDim)
    datarep_input = Input(shape=(None, None), name="true_data_rep")
    PRINT("True Data Representation Input:", datarep_input.shape)
    # Predicted Sequence Representation Input (BatchSize, SequenceLength, Code_EmbeddingDim)
    predrep_input = Input(shape=(None, None), name="pred_seq_rep")
    PRINT("Predicted Sequence Representation Input:", predrep_input.shape)
    # Concat Inputs
    disc_outputs = Concatenate(name="disc_concat")([datarep_input, predrep_input])
    PRINT("Discriminator Concat:", disc_outputs.shape)
    # Classifier
    # Flatten
    disc_outputs = Flatten(name="disc_flatten")(disc_outputs)
    PRINT("Discriminator Flatten:", disc_outputs.shape)
    # Dense Layers
    for i in range(len(dense_n_units)):
        disc_outputs = Dense(dense_n_units[i], activation=dense_activation, name=f"disc_dense_{i}")(disc_outputs)
        if dense_dropout > 0.0: disc_outputs = Dropout(dense_dropout, name=f"disc_dropout_{i}")(disc_outputs)
        PRINT(f"Discriminator Dense {i}:", disc_outputs.shape)
    # Final Dense Layer
    disc_outputs = Dense(units=1, activation="sigmoid", name="disc_dense")(disc_outputs)
    PRINT("Discriminator Dense:", disc_outputs.shape)
    # Model
    model = Model([datarep_input, predrep_input], disc_outputs)

    return model

def Discriminator_Token(
    datarep_dim=(100, 32), 
    token_rep_params={"type": "dense"}, 

    dense_n_units=[],
    dense_activation="relu",
    dense_dropout=0.0, 

    vocab_size=600,

    **params
    ):
    '''
    Discriminator - Token Type
    Inputs:
        - True Data Representation from Decoder Concat (BatchSize, ?) - Depends on Decoder Type
        - Predicted Token (BatchSize, VocabSize)
    Outputs:
        - Discriminator Output (BatchSize, 1)
    '''
    # True Data Representation Input (BatchSize, ?)
    datarep_input = Input(shape=datarep_dim, name="true_data_rep")
    PRINT("True Data Representation Input:", datarep_input.shape)
    ## Flatten True Data Representation Input (BatchSize, TrueFlatDim)
    datarep_final_input = Flatten(name="true_data_rep_flatten")(datarep_input)
    PRINT("True Data Representation Input Flatten:", datarep_final_input.shape)

    # Predicted Token Input (BatchSize, VocabSize)
    predtoken_input = Input(shape=(vocab_size,), name="pred_token")
    PRINT("Predicted Token Input:", predtoken_input.shape)
    # Get Token Representation (BatchSize, FinalTokenDim)
    token_rep = functools.partial(BLOCKS_TOKENREP[token_rep_params["type"]], **token_rep_params)
    token_rep_input = token_rep(predtoken_input)
    PRINT("Predicted Token Representation Input:", token_rep_input.shape)

    # Concat Inputs (BatchSize, TrueFlatDim + FinalTokenDim)
    disc_outputs = Concatenate(name="disc_concat")([datarep_final_input, token_rep_input])
    PRINT("Discriminator Concat:", disc_outputs.shape)

    # Classifier
    # Dense Layers
    for i in range(len(dense_n_units)):
        disc_outputs = Dense(dense_n_units[i], activation=dense_activation, name=f"disc_dense_{i}")(disc_outputs)
        if dense_dropout > 0.0: disc_outputs = Dropout(dense_dropout, name=f"disc_dropout_{i}")(disc_outputs)
        PRINT(f"Discriminator Dense {i}:", disc_outputs.shape)
    # Final Dense Layer
    disc_outputs = Dense(units=1, activation="sigmoid", name="disc_dense")(disc_outputs)
    PRINT("Discriminator Dense:", disc_outputs.shape)
    # Model
    model = Model([datarep_input, predtoken_input], disc_outputs)

    return model

# Main Vars
DISCRIMINATORS = {
    # "sequence": Discriminator_Sequence,
    "token": Discriminator_Token
}