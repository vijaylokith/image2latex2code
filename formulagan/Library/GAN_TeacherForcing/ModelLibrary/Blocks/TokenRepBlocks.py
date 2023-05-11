"""
Token Representation Blocks
"""

# Imports
from ..Utils import *

# Main Vars
PRINT = PRINT_FUNCS["token_combine"]

# Main Functions
def TokenRepBlock_Dense(
    token_input, 

    token_dense_units=[32],
    token_dense_activation="relu",
    token_dense_dropout=0.0,

    **params
    ):
    '''
    Token Representation Block - Dense
    Inputs:
        - Token (BatchSize, VocabSize)
    Architecture:
        - Pass through more dense layers to arrive at final output
    Outputs:
        - Token Representation (BatchSize, FinalTokenDim)
    '''
    # Init
    tokenrep_outputs = token_input
    # Token Dense Layers
    for i in range(len(token_dense_units)):
        tokenrep_outputs = Dense(token_dense_units[i], activation=token_dense_activation, name=f"combined_dense_{i}")(tokenrep_outputs)
        if token_dense_dropout > 0.0: tokenrep_outputs = Dropout(token_dense_dropout, name=f"combined_dropout_{i}")(tokenrep_outputs)
        PRINT(f"Token Rep Dense {i}:", tokenrep_outputs.shape)

    return tokenrep_outputs

# Main Vars
BLOCKS_TOKENREP = {
    "dense": TokenRepBlock_Dense
}