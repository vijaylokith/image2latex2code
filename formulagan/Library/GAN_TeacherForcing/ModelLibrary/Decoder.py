"""
Decoder

Note: Always keep concat layer name as "decoder_concat"
"""

# Imports
from .Blocks.DecoderBlocks import *

# Main Vars
PRINT = PRINT_FUNCS["decoder"]

# Main Functions
def Model_Decoder_Simple(
    encoder_output, 

    decoder_input_embedding_dim=64, 
    decoder_input_type="LSTM", 
    decoder_input_recurrent_units=32,
    decoder_input_dense_units=[32],
    decoder_input_dense_activation=None,
    decoder_input_dense_dropout=0.0,

    decoder_output_dense_units=[32, 64],
    decoder_output_dense_activation="relu",
    decoder_output_dense_dropout=0.0,

    vocab_size=600,
    max_sequence_length=200,

    **params
    ):
    '''
    Decoder - Simple
        - Input: Feature Vector + True Code Sequence
        - Output: Predicted Code Token
    '''
    # Decoder
    # Encoder Output Layer
    PRINT("Encoder Output (Image Feature Vector):", encoder_output.shape)
    # Input Layer
    decoder_input = Input(shape=(None,), name="decoder_input")
    PRINT("Decoder Input (Teacher Forcing):", decoder_input.shape)

    # Language Modelling Layer
    ## Embedding Layer
    decoder_embedding = Embedding(
        vocab_size, decoder_input_embedding_dim, 
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_input)
    PRINT("Decoder Embedding:", decoder_embedding.shape)
    decoder_outputs = decoder_embedding
    ## LSTM Layer
    decoderData = BLOCKS_DECODER[decoder_input_type](
        decoder_outputs, block_name="decoder_input_recurrent",
        n_units=decoder_input_recurrent_units,
        return_sequences=False, return_state=True,
        recurrent_initializer="glorot_uniform"
    )
    decoder_outputs = decoderData["output"]
    PRINT("Decoder Input Recurrent:", decoder_outputs.shape)
    ## Dense Layers
    for i in range(len(decoder_input_dense_units)):
        decoder_outputs = Dense(
            decoder_input_dense_units[i], activation=decoder_input_dense_activation, 
            name=f"decoder_input_dense_{i}"
        )(decoder_outputs)
        if decoder_input_dense_dropout > 0.0: decoder_outputs = Dropout(decoder_input_dense_dropout, name=f"decoder_input_dense_dropout_{i}")(decoder_outputs)
        PRINT(f"Decoder Input Dense {i}:", decoder_outputs.shape)

    # Concat Layer - Decoder Input + Encoder Output
    decoder_outputs = Concatenate(axis=-1, name="decoder_concat")([decoder_outputs, encoder_output])
    PRINT("Concat:", decoder_outputs.shape)

    # Decoder Output Layer
    ## Dense Layers
    for i in range(len(decoder_output_dense_units)):
        decoder_outputs = Dense(
            decoder_output_dense_units[i], activation=decoder_output_dense_activation,
            name=f"decoder_output_dense_{i}"
        )(decoder_outputs)
        if decoder_output_dense_dropout > 0.0: decoder_outputs = Dropout(decoder_output_dense_dropout, name=f"decoder_output_dense_dropout_{i}")(decoder_outputs)
        PRINT(f"Decoder Output Dense {i}:", decoder_outputs.shape)

    # Output Layer
    decoder_outputs = Dense(vocab_size, activation="softmax", name="decoder_dense")(
        decoder_outputs
    )
    PRINT("Decoder Output:", decoder_outputs.shape)

    decoder = {
        "input": decoder_input,
        "output": decoder_outputs
    }
    return decoder

def Model_Decoder_Recurrent(
    encoder_output, 

    decoder_input_embedding_dim=64, 
    decoder_input_type="LSTM",
    decoder_input_recurrent_units=32,
    decoder_input_recurrent_dense_units=32,
    decoder_input_recurrent_dense_activation=None,

    decoder_output_type="LSTM",
    decoder_output_recurrent_units=[32, 64], 
    decoder_output_recurrent_dense_units=[], 
    decoder_output_recurrent_dense_activation=None,
    decoder_output_dense_units=[],
    decoder_output_dense_activation="relu",
    decoder_output_dense_dropout=0.0,
    
    vocab_size=600,
    max_sequence_length=200,
    decoder_full_seq_output=True,

    **params
    ):
    '''
    Decoder - Recurrent
        - Input: Feature Vector + True Code Sequence
        - Output: Predicted Code Sequence / Predicted Code Token (depending on decoder_full_seq_output)
    '''
    # Decoder
    # Encoder Output Repeat Layer
    PRINT("Encoder Output (Image Feature Vector):", encoder_output.shape)
    encoder_output = RepeatVector(max_sequence_length, name="encoder_repeat")(encoder_output)
    PRINT("Encoder Output Repeated:", encoder_output.shape)
    # Input Layer
    decoder_input = Input(shape=(None,), name="decoder_input")
    PRINT("Decoder Input (Teacher Forcing):", decoder_input.shape)

    # Language Modelling Layer
    ## Embedding Layer
    decoder_embedding = Embedding(
        vocab_size, decoder_input_embedding_dim, 
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_input)
    PRINT("Decoder Embedding:", decoder_embedding.shape)
    decoder_outputs = decoder_embedding
    ## LSTM Layer
    decoderData = BLOCKS_DECODER[decoder_input_type](
        decoder_outputs, block_name="decoder_input_recurrent",
        n_units=decoder_input_recurrent_units,
        return_sequences=True, return_state=True,
        recurrent_initializer="glorot_uniform"
    )
    decoder_outputs = decoderData["output"]
    PRINT("Decoder Input Recurrent:", decoder_outputs.shape)
    ## Dense Layer
    if decoder_input_recurrent_dense_units > 0:
        decoder_outputs = TimeDistributed(
            Dense(decoder_input_recurrent_dense_units, activation=decoder_input_recurrent_dense_activation), 
            name="decoder_input_dense"
        )(decoder_outputs)
        PRINT("Decoder Input Dense:", decoder_outputs.shape)

    # Concat Layer - Decoder Input + Encoder Output
    decoder_outputs = Concatenate(axis=-1, name="decoder_concat")([decoder_outputs, encoder_output])
    PRINT("Concat:", decoder_outputs.shape)

    # Decoder Output Recurrent Layer
    ## LSTM + Dense Layers
    for i in range(len(decoder_output_recurrent_units)):
        # LSTM Layer
        return_sequences = True if decoder_full_seq_output else (i < (len(decoder_output_recurrent_units)-1))
        decoderData = BLOCKS_DECODER[decoder_output_type](
            decoder_outputs, block_name="decoder_output_recurrent_" + str(i),
            n_units=decoder_output_recurrent_units[i],
            return_state=True, return_sequences=return_sequences,
            recurrent_initializer="glorot_uniform"
        )
        decoder_outputs = decoderData["output"]
        PRINT(f"Decoder Output Recurrent {i}:", decoder_outputs.shape)
        # Dense Layer
        if return_sequences and (i < len(decoder_output_recurrent_dense_units)):
            decoder_outputs = TimeDistributed(
                Dense(decoder_output_recurrent_dense_units[i], activation=decoder_output_recurrent_dense_activation), 
                name=f"decoder_output_recurrent_dense_{i}"
            )(decoder_outputs)
            PRINT(f"Decoder Output Dense {i}:", decoder_outputs.shape)

    # Decoder Output Dense Layer
    ## Dense Layers
    for i in range(len(decoder_output_dense_units)):
        decoder_outputs = Dense(
            decoder_output_dense_units[i], activation=decoder_output_dense_activation,
            name=f"decoder_output_dense_{i}"
        )(decoder_outputs)
        if decoder_output_dense_dropout > 0.0: decoder_outputs = Dropout(decoder_output_dense_dropout, name=f"decoder_output_dense_dropout_{i}")(decoder_outputs)
        PRINT(f"Decoder Output Dense {i}:", decoder_outputs.shape)

    # Output Layer
    decoder_outputs = Dense(vocab_size, activation="softmax", name="decoder_dense")(
        decoder_outputs
    )
    PRINT("Decoder Output:", decoder_outputs.shape)

    decoder = {
        "input": decoder_input,
        "output": decoder_outputs
    }
    return decoder

def Model_Decoder_Attention(
    encoder_output, 

    decoder_input_embedding_dim=64, 
    decoder_input_type="LSTM", 
    decoder_input_recurrent_units=32,
    decoder_input_recurrent_dense_units=32,
    decoder_input_recurrent_dense_activation=None,

    decoder_output_type="LSTM",
    decoder_output_recurrent_units=[32, 64], 
    decoder_output_recurrent_dense_units=[], 
    decoder_output_recurrent_dense_activation=None,
    decoder_output_dense_units=[32, 64],
    decoder_output_dense_activation="relu",
    decoder_output_dense_dropout=0.0,

    vocab_size=600,
    max_sequence_length=200,
    decoder_full_seq_output=True,

    **params
    ):
    '''
    Decoder - Attention
        - Input: Feature Vector + True Code Sequence
        - Output: Predicted Code Token
    '''
    # Decoder
    # Encoder Output Layer
    PRINT("Encoder Output (Image Feature Vector):", encoder_output.shape)
    # Input Layer
    decoder_input = Input(shape=(None,), name="decoder_input")
    PRINT("Decoder Input (Teacher Forcing):", decoder_input.shape)

    # Language Modelling Layer
    ## Embedding Layer
    decoder_embedding = Embedding(
        vocab_size, decoder_input_embedding_dim, 
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_input)
    PRINT("Decoder Embedding:", decoder_embedding.shape)
    decoder_outputs = decoder_embedding
    ## LSTM Layer
    decoderData = BLOCKS_DECODER[decoder_input_type](
        decoder_outputs, block_name="decoder_input_recurrent",
        n_units=decoder_input_recurrent_units,
        return_sequences=True, return_state=True,
        recurrent_initializer="glorot_uniform"
    )
    decoder_outputs = decoderData["output"]
    PRINT("Decoder Input Recurrent:", decoder_outputs.shape)
    ## Dense Layer
    if decoder_input_recurrent_dense_units > 0:
        decoder_outputs = TimeDistributed(
            Dense(decoder_input_recurrent_dense_units, activation=decoder_input_recurrent_dense_activation), 
            name="decoder_input_dense"
        )(decoder_outputs)
        PRINT("Decoder Input Dense:", decoder_outputs.shape)
    ## Attention
    attn_outputs = Attention(name="decoder_attention")([decoder_outputs, encoder_output])
    PRINT("Attention:", attn_outputs.shape)

    # Concat Layer - Decoder Output + Attention Output
    decoder_outputs = Concatenate(axis=-1, name="decoder_concat")([decoder_outputs, attn_outputs])
    PRINT("Concat:", decoder_outputs.shape)

    # Decoder Output Recurrent Layer
    ## LSTM + Dense Layers
    for i in range(len(decoder_output_recurrent_units)):
        # LSTM Layer
        return_sequences = True if decoder_full_seq_output else (i < (len(decoder_output_recurrent_units)-1))
        decoderData = BLOCKS_DECODER[decoder_output_type](
            decoder_outputs, block_name="decoder_output_recurrent_" + str(i),
            n_units=decoder_output_recurrent_units[i],
            return_state=True, return_sequences=return_sequences,
            recurrent_initializer="glorot_uniform"
        )
        decoder_outputs = decoderData["output"]
        PRINT(f"Decoder Output Recurrent {i}:", decoder_outputs.shape)
        # Dense Layer
        if return_sequences and (i < len(decoder_output_recurrent_dense_units)):
            decoder_outputs = TimeDistributed(
                Dense(decoder_output_recurrent_dense_units[i], activation=decoder_output_recurrent_dense_activation), 
                name=f"decoder_output_recurrent_dense_{i}"
            )(decoder_outputs)
            PRINT(f"Decoder Output Dense {i}:", decoder_outputs.shape)
    # Decoder Output Dense Layer
    ## Dense Layers
    for i in range(len(decoder_output_dense_units)):
        decoder_outputs = Dense(
            decoder_output_dense_units[i], activation=decoder_output_dense_activation,
            name=f"decoder_output_dense_{i}"
        )(decoder_outputs)
        if decoder_output_dense_dropout > 0.0: decoder_outputs = Dropout(decoder_output_dense_dropout, name=f"decoder_output_dense_dropout_{i}")(decoder_outputs)
        PRINT(f"Decoder Output Dense {i}:", decoder_outputs.shape)
    # Output Layer
    decoder_outputs = Dense(vocab_size, activation="softmax", name="decoder_dense")(
        decoder_outputs
    )
    PRINT("Decoder Output:", decoder_outputs.shape)

    decoder = {
        "input": decoder_input,
        "output": decoder_outputs
    }
    return decoder

# Main Vars
DECODERS = {
    "token_output": {
        "simple": Model_Decoder_Simple,
        "recurrent": functools.partial(Model_Decoder_Recurrent, decoder_full_seq_output=False),
        "attention": functools.partial(Model_Decoder_Attention, decoder_full_seq_output=False),
    },
    "sequence_output": {
        "recurrent": functools.partial(Model_Decoder_Recurrent, decoder_full_seq_output=True),
        "attention": functools.partial(Model_Decoder_Attention, decoder_full_seq_output=True),
    }    
}
DECODERS_CONCAT_SHAPE = {
    "token_output": {
        "simple": lambda d, params: (params["encoder_output_shape"]+d["decoder_input_dense_units"][-1], ),
        "recurrent": lambda d, params: (params["max_sequence_length"], params["encoder_output_shape"]+d["decoder_input_recurrent_dense_units"]),
        "attention": lambda d, params: (params["max_sequence_length"], params["encoder_output_shape"]+d["decoder_input_recurrent_dense_units"]),
    },
    "sequence_output": {
        "recurrent": lambda d, params: (params["max_sequence_length"], params["encoder_output_shape"]+d["decoder_input_recurrent_dense_units"]),
        "attention": lambda d, params: (params["max_sequence_length"], params["encoder_output_shape"]+d["decoder_input_recurrent_dense_units"]),
    }
}