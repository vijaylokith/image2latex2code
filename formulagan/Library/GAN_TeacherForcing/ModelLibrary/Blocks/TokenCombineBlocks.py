"""
Token Combine Blocks
"""

# Imports
from ..Utils import *

# Main Vars
PRINT = PRINT_FUNCS["token_combine"]

# Main Functions
def TokenCombinerBlock_DenseConcat(
    predtoken_input, truetoken_input,

    predtoken_dense_units=[],
    predtoken_dense_activation="relu",
    predtoken_dense_dropout=0.0,
    truetoken_dense_units=[],
    truetoken_dense_activation="relu",
    truetoken_dense_dropout=0.0,

    combined_dense_units=[32],
    combined_dense_activation="relu",
    combined_dense_dropout=0.0,

    **params
    ):
    '''
    Token Combiner Block - Dense Concat
    Inputs:
        - Predicted Token (BatchSize, VocabSize)
        - True Token (BatchSize, VocabSize)
    Architecture:
        - Separate Dense Layers for both
        - Concatenate Dense Layers
        - Pass through more dense layers to arrive at final output
    Outputs:
        - Combined Representation (BatchSize, FinalTokenDim)
    '''
    # Predicted Token Dense Layers
    for i in range(len(predtoken_dense_units)):
        predtoken_input = Dense(predtoken_dense_units[i], activation=predtoken_dense_activation, name=f"predtoken_dense_{i}")(predtoken_input)
        if predtoken_dense_dropout > 0.0: predtoken_input = Dropout(predtoken_dense_dropout, name=f"predtoken_dropout_{i}")(predtoken_input)
        PRINT(f"Predicted Token Dense {i}:", predtoken_input.shape)
    # True Token Dense Layers
    for i in range(len(truetoken_dense_units)):
        truetoken_input = Dense(truetoken_dense_units[i], activation=truetoken_dense_activation, name=f"truetoken_dense_{i}")(truetoken_input)
        if truetoken_dense_dropout > 0.0: truetoken_input = Dropout(truetoken_dense_dropout, name=f"truetoken_dropout_{i}")(truetoken_input)
        PRINT(f"True Token Dense {i}:", truetoken_input.shape)
    # Concat Token Representations
    comb_outputs = Concatenate(name="combined_concat")([predtoken_input, truetoken_input])
    PRINT("Token Combiner Concat:", comb_outputs.shape)
    # Combined Dense Layers
    for i in range(len(combined_dense_units)):
        comb_outputs = Dense(combined_dense_units[i], activation=combined_dense_activation, name=f"combined_dense_{i}")(comb_outputs)
        if combined_dense_dropout > 0.0: comb_outputs = Dropout(combined_dense_dropout, name=f"combined_dropout_{i}")(comb_outputs)
        PRINT(f"Combined Dense {i}:", comb_outputs.shape)

    return comb_outputs

# Main Vars
BLOCKS_TOKENCOMBINE = {
    "dense_concat": TokenCombinerBlock_DenseConcat
}