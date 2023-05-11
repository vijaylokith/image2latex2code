"""
Encoder
"""

# Imports
from .Utils import *

# Main Vars
PRINT = PRINT_FUNCS["encoder"]

# Main Functions
# Encoder Functions
def Model_Encoder_SimpleCNN(
    input_shape=(224, 224),
    output_shape=512,
    output_activation="relu", 

    conv_n_filters=[32, 64],
    conv_activation="relu",
    conv_dropout=0.0,
    
    dense_n_units=[512],
    dense_activation="relu",
    dense_dropout=0.0,

    **params
    ):
    '''
    Encoder - Simple - Conv-Maxpool Layers + Flatten + Dense Layers
        - Input: Image
        - Output: Feature Vector
    '''
    # Input Layer
    encoder_input = Input(shape=input_shape, name="encoder_input")
    PRINT("Encoder Input:", encoder_input.shape)
    encoder_outputs = encoder_input
    # Conv Layers
    for i in range(len(conv_n_filters)):
        encoder_outputs = Conv2D(conv_n_filters[i], (3, 3), activation=conv_activation, padding="same", name="encoder_conv_"+str(i))(
            encoder_outputs
        )
        encoder_outputs = MaxPooling2D((2, 2), name="encoder_maxpool_"+str(i))(
            encoder_outputs
        )
        if conv_dropout > 0.0: encoder_outputs = Dropout(conv_dropout, name="encoder_dropout_"+str(i))(encoder_outputs)
        PRINT(f"Encoder Conv {i}:", encoder_outputs.shape)
    # Dense Layers
    encoder_outputs = Flatten(name="flatten")(encoder_outputs)
    PRINT("Encoder Flatten:", encoder_outputs.shape)
    for i in range(len(dense_n_units)):
        encoder_outputs = Dense(dense_n_units[i], activation=dense_activation, name="encoder_dense_"+str(i))(encoder_outputs)
        if dense_dropout > 0.0: encoder_outputs = Dropout(dense_dropout, name="encoder_dense_dropout_"+str(i))(encoder_outputs)
        PRINT(f"Encoder Dense {i}:", encoder_outputs.shape)
    # Output Layer
    encoder_outputs = Dense(output_shape, activation=output_activation, name="encoder_output_dense")(encoder_outputs)
    PRINT("Encoder Output:", encoder_outputs.shape)

    encoder = {
        "input": encoder_input,
        "output": encoder_outputs
    }
    return encoder

def Model_Encoder_AttentionCNN(
    input_shape=(224, 224),
    output_shape=512,
    output_activation="relu", 

    conv_n_filters=[32, 64],
    conv_activation="relu",
    conv_dropout=0.0,
    
    dense_n_units=[512],
    dense_activation="relu",
    dense_dropout=0.0,

    **params
    ):
    '''
    Encoder - Attention - Conv-Maxpool Layers + Dense Layers
        - Input: Image
        - Output: Image Features
    '''
    # Input Layer
    encoder_input = Input(shape=input_shape, name="encoder_input")
    PRINT("Encoder Input:", encoder_input.shape)
    encoder_outputs = encoder_input
    # Conv Layers
    for i in range(len(conv_n_filters)):
        encoder_outputs = Conv2D(conv_n_filters[i], (3, 3), activation=conv_activation, padding="same", name="encoder_conv_"+str(i))(
            encoder_outputs
        )
        encoder_outputs = MaxPooling2D((2, 2), name="encoder_maxpool_"+str(i))(
            encoder_outputs
        )
        if conv_dropout > 0.0: encoder_outputs = Dropout(conv_dropout, name="encoder_dropout_"+str(i))(encoder_outputs)
        PRINT(f"Encoder Conv {i}:", encoder_outputs.shape)
    # Flatten Non-Filter Dimensions
    encoder_outputs = Reshape((-1, encoder_outputs.shape[-1]), name="encoder_reshape")(encoder_outputs)
    # Dense Layers
    for i in range(len(dense_n_units)):
        encoder_outputs = Dense(dense_n_units[i], activation=dense_activation, name="encoder_dense_"+str(i))(encoder_outputs)
        if dense_dropout > 0.0: encoder_outputs = Dropout(dense_dropout, name="encoder_dense_dropout_"+str(i))(encoder_outputs)
        PRINT(f"Encoder Dense {i}:", encoder_outputs.shape)
    # Output Layer
    encoder_outputs = Dense(output_shape, activation=output_activation, name="encoder_output_dense")(encoder_outputs)
    PRINT("Encoder Output:", encoder_outputs.shape)

    encoder = {
        "input": encoder_input,
        "output": encoder_outputs
    }
    return encoder

# Main Vars
ENCODERS = {
    "simple": Model_Encoder_SimpleCNN,
    "attention": Model_Encoder_AttentionCNN
}