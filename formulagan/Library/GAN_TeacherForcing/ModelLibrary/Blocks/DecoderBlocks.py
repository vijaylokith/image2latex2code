'''
Decoder Block Functions
'''

# Imports
from ..Utils import *

# Main Functions
# Decoder Block Functions
# RNN Block
def DecoderBlock_RNN(
    model, 
    n_units=256, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    Decoder Block - RNN Block
    '''
    decoder = SimpleRNN(
        n_units, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_RNN", 
        **params
    )
    data = decoder(model)
    decoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return decoder_data

# LSTM Block
def DecoderBlock_LSTM(
    model, 
    n_units=256, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    Decoder Block - LSTM Block
    '''
    decoder = LSTM(
        n_units, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_LSTM", 
        **params
    )
    data = decoder(model)
    decoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return decoder_data

# GRU Block
def DecoderBlock_GRU(
    model, 
    n_units=256, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    Decoder Block - GRU Block
    '''
    decoder = GRU(
        n_units, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_GRU", 
        **params
    )
    data = decoder(model)
    decoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return decoder_data
    
# Main Vars
BLOCKS_DECODER = {
    "RNN": DecoderBlock_RNN,
    "LSTM": DecoderBlock_LSTM,
    "GRU": DecoderBlock_GRU
}