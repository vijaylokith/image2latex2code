"""
Dataset

IM2LATEX_100K Dataset
X : Images - (BatchSize, Height, Width, 1)
Y : LaTeX Code - (BatchSize, SequenceMaxSize, VocabSize)
"""

# Imports
from .DatasetUtils import *
from .ModelLibrary.DatasetGenerators import *

# Main Vars
DATASET_PARAMS = {
    "images": {
        "n_channels": 1,
        "size": (54, 256, 1)
    },
    "codes": {
        "max_sequence_length": 100,
        "max_vocab_size": 567
    }
}

# Main Functions
# Load Functions
def Image_ResizeWithPadding(I, target_size, padding_val=255.0):
    '''
    Image - Resize Image with Padding
    '''
    # Init
    I = np.array(I)
    # Black Padding
    # I = tf.image.resize_with_pad(I, DATASET_PARAMS["images"]["size"][0], DATASET_PARAMS["images"]["size"][1])
    # Value Padding Init
    I_padded = np.ones(
        (DATASET_PARAMS["images"]["size"][0], DATASET_PARAMS["images"]["size"][1], DATASET_PARAMS["images"]["n_channels"]), 
        dtype=float
    ) * padding_val
    # Identify Padding Dimension
    target_aspect_ratio = target_size[0] / target_size[1]
    I_aspect_ratio = I.shape[0] / I.shape[1]
    # Aspect Ratio = Height / Width
    # If Image Aspect Ratio > Target Aspect Ratio - For same height, image width is less than target width (can pad)
    # Resize to fit height and pad width
    if I_aspect_ratio > target_aspect_ratio:
        # Resize to fit height preserving image aspect ratio
        new_w = int(target_size[0]/I_aspect_ratio)
        I_resized = tf.image.resize(I, (target_size[0], new_w))
        # Put Image in Center of Padded Image
        offset_w = (target_size[1] - I_resized.shape[1]) // 2
        I_padded[:, offset_w:offset_w+I_resized.shape[1], :] = I_resized
    # If Image Aspect Ratio <= Target Aspect Ratio - For same width, image height is less than target height (can pad)
    # Resize to fit width and pad height
    else:
        # Resize to fit width preserving image aspect ratio
        new_h = int(target_size[1]*I_aspect_ratio)
        I_resized = tf.image.resize(I, (new_h, target_size[1]))
        # Put Image in Center of Padded Image
        offset_h = (target_size[0] - I_resized.shape[0]) // 2
        I_padded[offset_h:offset_h+I_resized.shape[0], :, :] = I_resized
    I_padded = tf.convert_to_tensor(I_padded, dtype=tf.float32)

    return I_padded

def Dataset_LoadImage_Tensorflow(path):
    '''
    Dataset - Load Image as Tensorflow Tensor
    '''
    # Read and Decode Image
    I = tf.io.read_file(path)
    I = tf.io.decode_jpeg(I, channels=DATASET_PARAMS["images"]["n_channels"])
    # Apply Padding
    I = Image_ResizeWithPadding(I, DATASET_PARAMS["images"]["size"], padding_val=255.0)
    # Resize without Crop Image
    # I = Resizing(DATASET_PARAMS["images"]["size"][0], DATASET_PARAMS["images"]["size"][1])(I)
    # Resize with Crop Image
    # I = Resizing(DATASET_PARAMS["images"]["size"][0], DATASET_PARAMS["images"]["size"][1], crop_to_aspect_ratio=True)(I)
    
    return I

def Dataset_LoadCode_Split(code, tokenizer, lookups):
    '''
    Dataset - Load Code - Split into Sequence and Final Token
    '''
    code = code.decode("utf-8")
    tokens = code.split(" ")
    inp = tokenizer(" ".join(tokens[:-1]))
    out = lookups["word_to_index"](tokens[-1])

    return inp, out

# Tokenizer Functions
def Dataset_Tokenizer_GetLookups(tokenizer):
    '''
    Tokenizer - Get (Word -> Index) and (Index -> Word) Lookups
    '''
    word_to_index = StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary()
    )
    index_to_word = StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True
    )

    LookUps = {
        "word_to_index": word_to_index,
        "index_to_word": index_to_word
    }
    return LookUps

def Dataset_Tokenizer_GetTokenizer(
    codes, 
    max_sequence_length=DATASET_PARAMS["codes"]["max_sequence_length"], 
    **params
    ):
    '''
    Tokenizer - Create Tokenizer
    '''
    codes = list(codes)
    for i in range(len(codes)):
        codes[i] = f"<start> {codes[i]} <end>"
    # Create Tensorflow Dataset
    DATASET_TF = tf.data.Dataset.from_tensor_slices(codes)
    # Create Tokenizer
    TOKENIZER = TextVectorization(
        standardize=None,
        output_sequence_length=max_sequence_length
    )
    # Learn Vocab from Dataset
    TOKENIZER.adapt(DATASET_TF)

    return TOKENIZER

def Dataset_Tokenizer_SaveTokenizer(tokenizer, path):
    '''
    Tokenizer - Save Tokenizer as model
    '''
    tokenizer_model = Sequential()
    tokenizer_model.add(Input(shape=(1,), dtype=tf.string))
    tokenizer_model.add(tokenizer)
    tokenizer_model.save(path, save_format="tf")

def Dataset_Tokenizer_LoadTokenizer(path):
    '''
    Tokenizer - Load Tokenizer from model
    '''
    tokenizer_model = load_model(path)
    tokenizer = tokenizer_model.layers[-1]

    return tokenizer

# Dataset Threshold Functions
def DatasetDF_ThresholdCodeLength(DATASET_df, max_sequence_length=DATASET_PARAMS["codes"]["max_sequence_length"]):
    '''
    DatasetDF - Threshold Code Size and drop rows with large lengths
    '''
    CodeLengths = DATASET_df["code"].str.split(" ").apply(len)
    DATASET_df = DATASET_df.drop(DATASET_df[CodeLengths > max_sequence_length].index)
    DATASET_df.reset_index(drop=True, inplace=True)

    return DATASET_df

# Preprocess Functions
def Dataset_PreprocessImageData(
    paths, 
    **params
    ):
    '''
    Dataset - Preprocess Image Data
    '''
    DATASET_TF = list(paths)

    return DATASET_TF

def Dataset_PreprocessCodeData(
    codes, 
    TOKENIZER=None, 
    max_sequence_length=DATASET_PARAMS["codes"]["max_sequence_length"], 
    expand_sequences=False,
    **params
    ):
    '''
    Dataset - Preprocess Code Data
    '''
    # Add Start and End Tokens and Collect Sequence Lengths
    codes = list(codes)
    codesData = {
        "sequence_lengths": [],
        "total_sequence_length": 0
    }
    for i in range(len(codes)):
        seq = f"<start> {codes[i]} <end>"
        codes[i] = seq
        codesData["sequence_lengths"].append(len(seq.split(" ")))
        codesData["total_sequence_length"] += codesData["sequence_lengths"][-1]
    # Create Tensorflow Dataset
    DATASET_TF = tf.data.Dataset.from_tensor_slices(codes)
    # Create Tokenizer if not provided
    if TOKENIZER is None:
        TOKENIZER = TextVectorization(
            standardize=None,
            output_sequence_length=max_sequence_length
        )
        # Learn Vocab from Dataset
        TOKENIZER.adapt(DATASET_TF)
    codesData["lookups"] = Dataset_Tokenizer_GetLookups(TOKENIZER)

    # Expand Sequences
    if expand_sequences:
        codes_expanded = []
        for i in range(len(codes)):
            seq = codes[i].split(" ")
            seq_expanded = []
            for j in range(1, len(seq)):
                seq_expanded.append(" ".join(seq[:j+1]))
            codes_expanded.extend(seq_expanded)
        DATASET_TF = tf.data.Dataset.from_tensor_slices(codes_expanded)
        # Create the tokenized vectors
        load_func = functools.partial(Dataset_LoadCode_Split, tokenizer=TOKENIZER, lookups=codesData["lookups"])
        DATASET_TF = DATASET_TF.map(lambda x: tf.numpy_function(load_func, [x], [tf.int64, tf.int64]))
    else:
        # Create the tokenized vectors
        DATASET_TF = DATASET_TF.map(lambda x: TOKENIZER(x))

    DATA = {
        "dataset": DATASET_TF,
        "tokenizer": TOKENIZER
    }
    DATA.update(codesData)

    return DATA

# Dataset Create Functions
def Dataset_GetDataset(
    path, tokenizer=None, mode="train", 
    expand_sequences=False,
    batch_size=32, buffer_size=1024, 
    N=-1, 
    **params
    ):
    '''
    Dataset - Get Tensorflow Dataset of Test / Val
    '''
    DATASET = {mode: {}}
    # Load Dataset
    DATASET_df = DATASET_FUNCS[mode](path, N=N)
    DATASET_df.dropna(inplace=True)
    DATASET_df.reset_index(drop=True, inplace=True)
    # Threshold Code Length
    DATASET_df = DatasetDF_ThresholdCodeLength(DATASET_df, max_sequence_length=DATASET_PARAMS["codes"]["max_sequence_length"])

    # Preprocess Images
    IMAGES = Dataset_PreprocessImageData(DATASET_df["path"])
    # Preprocess Codes
    CodesData = Dataset_PreprocessCodeData(
        DATASET_df["code"], TOKENIZER=tokenizer, 
        max_sequence_length=DATASET_PARAMS["codes"]["max_sequence_length"], 
        expand_sequences=expand_sequences
    )
    CODES, DATASET["tokenizer"] = CodesData["dataset"], CodesData["tokenizer"]
    DATASET["lookups"] = CodesData["lookups"]

    # Expand Sequences if needed
    if expand_sequences:
        IMAGES_EXPANDED = []
        for i in range(len(CodesData["sequence_lengths"])):
            count = CodesData["sequence_lengths"][i]
            IMAGES_EXPANDED.extend([IMAGES[i] for _ in range(count)])
        IMAGES = IMAGES_EXPANDED

    # Create Tensorflow Dataset - Images
    dataset_images = tf.data.Dataset.from_tensor_slices(IMAGES)
    dataset_images = dataset_images.map(
        lambda p: tf.numpy_function(Dataset_LoadImage_Tensorflow, [p], [tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Shuffle and Batch
    # dataset_images = dataset_images.shuffle(buffer_size)
    dataset_images = dataset_images.batch(batch_size)
    dataset_images = dataset_images.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Create Tensorflow Dataset - Codes
    dataset_codes = CODES
    # Shuffle and Batch
    # dataset_codes = dataset_codes.shuffle(buffer_size)
    dataset_codes = dataset_codes.batch(batch_size)
    dataset_codes = dataset_codes.prefetch(buffer_size=tf.data.AUTOTUNE)

    DATASET[mode]["dataset"] = {
        "images": dataset_images,
        "codes": dataset_codes
    }
    seq_mode = "teacher_forcing" if expand_sequences else "simple"
    DATASET[mode]["n_samples"] = DATASET_df.shape[0]
    DATASET[mode]["batch_size"] = batch_size
    DATASET[mode]["total_sequence_length"] = CodesData["total_sequence_length"]
    DATASET[mode]["n_steps"] = DATASETGEN_SIZES[seq_mode](DATASET[mode]) // batch_size

    return DATASET

# Driver Code
# I_path = "Data/InputImages/GroupTest_6.PNG"
# DATASET_PARAMS["images"]["size"] = (54, 256)
# I = Dataset_LoadImage_Tensorflow(I_path)
# I_original = cv2.imread(I_path, 0)
# print("Original:", I_original.shape)
# print("Resized:", I.shape[:2])
# plt.subplot(1, 2, 1)
# plt.imshow(I_original, "gray")
# plt.title(f"Original {I_original.shape}")
# plt.subplot(1, 2, 2)
# plt.imshow(I[:, :, 0], "gray")
# plt.title(f"Resized {I.shape[:2]}")
# plt.show()