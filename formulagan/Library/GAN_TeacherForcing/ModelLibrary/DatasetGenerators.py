"""
Dataset Generators
"""

# Imports
from .Utils import *

# Main Functions
# DataGen Utils Functions
def DataGenUtils_GetNormFunc(norm=True, norm_invert=False):
    '''
    DataGen Utils - Get Normaliser Function
    '''
    if not norm: return lambda x: x
    elif norm_invert: return lambda x: (255.0 - x) / 255.0
    else: return lambda x: x / 255.0

def DataGenUtils_GetOneHotFunc(vocab_size=600):
    '''
    DataGen Utils - Get One-Hot Function
    '''
    return lambda x: tf.one_hot(x, vocab_size)

# DataGen Size Functions
def DataGen_Size_Simple(DATASET):
    '''
    DataGen Size - Simple
    '''
    return DATASET["n_samples"]

def DataGen_Size_TeacherForcing(DATASET):
    '''
    DataGen Size - Teacher Forcing
    '''
    return DATASET["total_sequence_length"] - DATASET["n_samples"]

# DataGen Functions
def DataGen_Simple(
    Is, Cs, 
    epochs=1, vocab_size=600, 
    norm=False, norm_invert=False,
    one_hot=[False, False]
    ):
    '''
    Data Generator - Simple
    Restrictions:
        - Works only with "Entire Sequence Predictor" type Decoder ("sequence_output")
    Inputs:
        - Is: Iterator for Images
            - every iteration, returns I (BatchSize, Height, Width, Channels)
        - Cs: Iterator for Codes
            - every iteration, returns C (BatchSize, SequenceLength)
    Outputs:
        - (enc_inp, dec_inp), dec_out
            - enc_inp: Input to Encoder (BatchSize, Height, Width, Channels)
            - dec_inp: Input to Decoder (BatchSize, SequenceLength, ?VocabSize)
            - dec_out: True Expected Output from Decoder (BatchSize, SequenceLength, ?VocabSize)
    '''
    # Funcs
    norm_func = DataGenUtils_GetNormFunc(norm, norm_invert)
    one_hot_func = DataGenUtils_GetOneHotFunc(vocab_size)
    one_hot_present = one_hot[0] or one_hot[1]

    # Create Generator
    # Epoch Loop
    for epoch in range(epochs):
        # Batch Loop
        for I, C in zip(Is, Cs):
            I = norm_func(I[0])
            if one_hot_present: C_oh = one_hot_func(C)
            
            enc_inp = I
            dec_inp = C_oh if one_hot[0] else C
            dec_out = C_oh if one_hot[1] else C

            # Fix Batch Sizes if mismatched
            if enc_inp.shape[0] != dec_inp.shape[0] or enc_inp.shape[0] != dec_out.shape[0] or dec_inp.shape[0] != dec_out.shape[0]:
                minBatchSize = min(enc_inp.shape[0], dec_inp.shape[0], dec_out.shape[0])
                enc_inp = enc_inp[:minBatchSize]
                dec_inp = dec_inp[:minBatchSize]
                dec_out = dec_out[:minBatchSize]

            yield [enc_inp, dec_inp], dec_out

def DataGen_TeacherForcing(
    Is, Cs, 
    epochs=1, vocab_size=600, 
    norm=False, norm_invert=False,
    one_hot=[False, False]
    ):
    '''
    Data Generator - Teacher Forcing
    Restrictions:
        - Works only with "Next Token Predictor" type Decoder ("token_output")
    Inputs:
        - Is: Iterator for Images
            - every iteration, returns I (BatchSize, Height, Width, Channels)
        - Cs: Iterator for Codes
            - every iteration, returns [C_in (BatchSize, SequenceLength), C_out (BatchSize, 1)]
    Outputs:
        - (enc_inp, dec_inp), dec_out
            - enc_inp: Input to Encoder (BatchSize, Height, Width, Channels)
            - dec_inp: Input to Decoder (BatchSize, SequenceLength, ?VocabSize)
            - dec_out: True Expected Output from Decoder (BatchSize, ?VocabSize)
    '''
    # Funcs
    norm_func = DataGenUtils_GetNormFunc(norm, norm_invert)
    one_hot_func = DataGenUtils_GetOneHotFunc(vocab_size)

    # Create Generator
    # Epoch Loop
    for epoch in range(epochs):
        # Batch-Seq Loop
        for I, C in zip(Is, Cs):
            I = norm_func(I[0])
            C_in, C_out = C

            enc_inp = I
            dec_inp = one_hot_func(C_in) if one_hot[0] else C_in
            dec_out = one_hot_func(C_out) if one_hot[1] else C_out

            # Fix Batch Sizes if mismatched
            if enc_inp.shape[0] != dec_inp.shape[0] or enc_inp.shape[0] != dec_out.shape[0] or dec_inp.shape[0] != dec_out.shape[0]:
                minBatchSize = min(enc_inp.shape[0], dec_inp.shape[0], dec_out.shape[0])
                enc_inp = enc_inp[:minBatchSize]
                dec_inp = dec_inp[:minBatchSize]
                dec_out = dec_out[:minBatchSize]

            yield [enc_inp, dec_inp], dec_out

def DataGen_Simple2TeacherForcing(
    Is, Cs, 
    epochs=1, vocab_size=600, 
    norm=False, norm_invert=False,
    one_hot=[False, False]
    ):
    '''
    Data Generator - Teacher Forcing
    Restrictions:
        - Works only with "Next Token Predictor" type Decoder ("token_output")
    Inputs:
        - Is: Iterator for Images
            - every iteration, returns I (BatchSize, Height, Width, Channels)
        - Cs: Iterator for Codes
            - every iteration, returns C (BatchSize, SequenceLength)
    Outputs:
        - (enc_inp, dec_inp), dec_out
            - enc_inp: Input to Encoder (BatchSize, Height, Width, Channels)
            - dec_inp: Input to Decoder (BatchSize, SequenceLength, ?VocabSize)
            - dec_out: True Expected Output from Decoder (BatchSize, ?VocabSize)
    '''
    # Funcs
    norm_func = DataGenUtils_GetNormFunc(norm, norm_invert)
    one_hot_func = DataGenUtils_GetOneHotFunc(vocab_size)

    # Create Generator
    # Epoch Loop
    for epoch in range(epochs):
        # Batch-Seq Loop
        for I, C in zip(Is, Cs):
            I = norm_func(I[0])
            # C is a sequence of tokens with padded zeros => (a1, a2, ... an-1, an, 0, ... 0)
            # C_in => ((a1, 0, ...), (a1, a2, 0, ...), ... (a1, a2, ... an-1, 0, ...))
            # C_out => (a2, a3, ... an)
            I_temp = []
            C_in = []
            C_out = []
            C_lengths = tf.math.count_nonzero(C, axis=-1)
            for i in range(C.shape[0]):
                for j in range(1, C_lengths[i]):
                    cin = np.zeros(C.shape[1])
                    cin[:j] = C[i, :j]
                    C_in.append(tf.convert_to_tensor(cin))
                    C_out.append(C[i, j])
                    I_temp.append(I[i])
            C_in = tf.stack(C_in)
            C_out = tf.stack(C_out)
            I = tf.stack(I_temp)

            enc_inp = I
            dec_inp = one_hot_func(C_in) if one_hot[0] else C_in
            dec_out = one_hot_func(C_out) if one_hot[1] else C_out

            yield [enc_inp, dec_inp], dec_out

def DataGen_GANTeacherForcing(
    Is, Cs, 
    epochs=1, vocab_size=600, 
    norm=False, norm_invert=False,
    one_hot=[False, False, False]
    ):
    '''
    Data Generator - GAN Teacher Forcing
    Restrictions:
        - Works only with "Next Token Predictor" type Decoder ("token_output")
    Inputs:
        - Is: Iterator for Images
            - every iteration, returns I (BatchSize, Height, Width, Channels)
        - Cs: Iterator for Codes
            - every iteration, returns C (BatchSize, SequenceLength)
    Outputs:
        - (enc_inp, dec_inp), dec_out, y_true
            - enc_inp: Input to Encoder (BatchSize, Height, Width, Channels)
            - dec_inp: Input to Decoder (BatchSize, SequenceLength, ?VocabSize)
            - dec_out: True Expected Output from Decoder (BatchSize, ?VocabSize)
            - y_true: True Expected Output from Decoder (BatchSize, ?VocabSize)
    '''
    # Funcs
    norm_func = DataGenUtils_GetNormFunc(norm, norm_invert)
    one_hot_func = DataGenUtils_GetOneHotFunc(vocab_size)

    # Create Generator
    # Epoch Loop
    for epoch in range(epochs):
        # Batch-Seq Loop
        for I, C in zip(Is, Cs):
            I = norm_func(I[0])
            # C is a sequence of tokens with padded zeros => (a1, a2, ... an-1, an, 0, ... 0)
            # C_in => ((a1, 0, ...), (a1, a2, 0, ...), ... (a1, a2, ... an-1, 0, ...))
            # C_out => (a2, a3, ... an)
            I_temp = []
            C_in = []
            C_out = []
            C_lengths = tf.math.count_nonzero(C, axis=-1)
            for i in range(C.shape[0]):
                for j in range(1, C_lengths[i]):
                    cin = np.zeros(C.shape[1])
                    cin[:j] = C[i, :j]
                    C_in.append(tf.convert_to_tensor(cin))
                    C_out.append(C[i, j])
                    I_temp.append(I[i])
            C_in = tf.stack(C_in)
            C_out = tf.stack(C_out)
            I = tf.stack(I_temp)

            enc_inp = I
            dec_inp = one_hot_func(C_in) if one_hot[0] else C_in
            dec_out = one_hot_func(C_out) if one_hot[1] else C_out
            y_true = one_hot_func(C) if one_hot[2] else C

            yield [enc_inp, dec_inp], dec_out, y_true

def DataGen_GANTeacherForcing_Discriminator(
    Is, Cs, 
    epochs=1, vocab_size=600, 
    norm=False, norm_invert=False,
    one_hot=[False]
    ):
    '''
    Data Generator - GAN Teacher Forcing for Discriminator
    Restrictions:
        - Works only with "Next Token Predictor" type Decoder ("token_output")
    Inputs:
        - Is: Iterator for Images
            - every iteration, returns I (BatchSize, Height, Width, Channels)
        - Cs: Iterator for Codes
            - every iteration, returns C (BatchSize, SequenceLength)
    Outputs:
        - (enc_inp, dec_inp), gan_out
            - enc_inp: Input to Encoder (BatchSize, Height, Width, Channels)
            - dec_inp: Input to Decoder (BatchSize, SequenceLength, ?VocabSize)
            - gan_out: True Expected Output from GAN Discriminator (BatchSize, 1) => Always all 1.0
    '''
    # Funcs
    norm_func = DataGenUtils_GetNormFunc(norm, norm_invert)
    one_hot_func = DataGenUtils_GetOneHotFunc(vocab_size)

    # Create Generator
    # Epoch Loop
    for epoch in range(epochs):
        # Batch-Seq Loop
        for I, C in zip(Is, Cs):
            I = norm_func(I[0])
            # C is a sequence of tokens with padded zeros => (a1, a2, ... an-1, an, 0, ... 0)
            # C_in => ((a1, 0, ...), (a1, a2, 0, ...), ... (a1, a2, ... an-1, 0, ...))
            # C_out => (a2, a3, ... an) - NOT NEEDED FOR DISCRIMINATOR
            I_temp = []
            C_in = []
            C_lengths = tf.math.count_nonzero(C, axis=-1)
            for i in range(C.shape[0]):
                for j in range(1, C_lengths[i]):
                    cin = np.zeros(C.shape[1])
                    cin[:j] = C[i, :j]
                    C_in.append(tf.convert_to_tensor(cin))
                    I_temp.append(I[i])
            C_in = tf.stack(C_in)
            I = tf.stack(I_temp)

            enc_inp = I
            dec_inp = one_hot_func(C_in) if one_hot[0] else C_in
            gan_out = tf.zeros((enc_inp.shape[0], 1), dtype=tf.float32)

            yield [enc_inp, dec_inp], gan_out

# Main Vars
DATASETGEN_FUNCS = {
    "simple": DataGen_Simple,
    "teacher_forcing": DataGen_TeacherForcing,
    "simple_teacher_forcing": DataGen_Simple2TeacherForcing,
    "gan_teacher_forcing": DataGen_GANTeacherForcing,
    "gan_teacher_forcing_discriminator": DataGen_GANTeacherForcing_Discriminator
}
DATASETGEN_SIZES = {
    "simple": DataGen_Size_Simple,
    "teacher_forcing": DataGen_Size_TeacherForcing,
    "simple_teacher_forcing": DataGen_Size_Simple,
    "gan_teacher_forcing": DataGen_Size_Simple,
    "gan_teacher_forcing_discriminator": DataGen_Size_Simple
}