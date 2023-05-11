"""
Streamlit App
"""

# Imports
import os
import time
import streamlit as st

from FormulaOCR import *
from Utils.ImageUtils import *

# Main Vars
DEFAULT_IMAGE_PATH = "Data/InputImages/Test.PNG"
TEMP_PATH = "Data/Temp/"
FORMULAGAN_MODELS_DIR = "Models/FormulaGAN/"

# UI Functions
def UI_LoadModel_FormulaGAN():
    '''
    Load FormulaGAN Model
    '''
    # Select Model
    MODEL_PATHS = list(os.listdir(FORMULAGAN_MODELS_DIR))
    USERINPUT_ModelPath = st.sidebar.selectbox("Select FormulaGAN Model", MODEL_PATHS)
    USERINPUT_ModelPath = os.path.join(FORMULAGAN_MODELS_DIR, USERINPUT_ModelPath)
    # Load Model
    FormulaOCR_GAN_TeacherForcing.OCR_PATHS["model"] = USERINPUT_ModelPath
    FormulaOCR_GAN_TeacherForcing.OCR["model"] = FormulaOCR_GAN_TeacherForcing.Model_LoadModel(USERINPUT_ModelPath)

def UI_LoadDataset():
    '''
    Load Dataset
    '''
    st.markdown("## Load Dataset")
    # Select Dataset
    USERINPUT_Dataset = st.selectbox("Select Dataset", list(DATASETS.keys()))
    DATASET_MODULE = DATASETS[USERINPUT_Dataset]
    # Load Dataset
    DATASET = DATASET_MODULE.DATASET_FUNCS["test"]()
    N = DATASET.shape[0]

    # Subset Dataset
    st.markdown("## Subset Dataset")
    col1, col2 = st.columns(2)
    USERINPUT_DatasetStart = col1.number_input("Subset Dataset (Start Index)", 0, N-1, 0)
    USERINPUT_DatasetCount = col2.number_input("Subset Dataset (Count)", 1, N, N)
    USERINPUT_DatasetCount = min(USERINPUT_DatasetCount, N-USERINPUT_DatasetStart)
    DATASET = DATASET.iloc[USERINPUT_DatasetStart:USERINPUT_DatasetStart+USERINPUT_DatasetCount]
    DATASET.reset_index(drop=True, inplace=True)

    # Display Top N Images
    N = DATASET.shape[0]
    USERINPUT_ViewSampleIndex = st.slider(f"View Sample ({N} Samples)", 0, N-1, 0, 1)
    st.image(DATASET["path"][USERINPUT_ViewSampleIndex], caption=f"Image: {USERINPUT_ViewSampleIndex}", use_column_width=True)
    st.latex(DATASET["code"][USERINPUT_ViewSampleIndex])

    return DATASET

def UI_LoadLaTeXCode():
    '''
    Load LaTeX Code
    '''
    # Read
    st.markdown("## Load LaTeX Code")
    USERINPUT_LaTeXCode = st.text_area("LaTeX Code", "E = mc^2")
    # Display
    st.markdown("```latex\n" + USERINPUT_LaTeXCode + "\n```")
    st.latex(USERINPUT_LaTeXCode)

    return USERINPUT_LaTeXCode

def UI_GenerateLaTeXImage():
    '''
    Generate LaTeX Image
    '''
    # Params
    USERINPUT_Code = UI_LoadLaTeXCode()
    USERINPUT_Euler = st.checkbox("Euler")
    USERINPUT_DPI = st.number_input("DPI", 1, 5000, 500, 1)

    # Convert to Image
    USERINPUT_Image = DATASETS["GeneratedDatasets"].DatasetUtils_LaTeX2Image(USERINPUT_Code, euler=USERINPUT_Euler, dpi=USERINPUT_DPI)

    return USERINPUT_Image

def UI_LoadImage():
    '''
    Load Image
    '''
    # Load Image
    st.markdown("## Load Image")
    USERINPUT_LoadType = st.selectbox("Load Type", ["Examples", "Upload File", "Datasets", "Generate"])
    if USERINPUT_LoadType == "Examples":
        # Load Filenames from Default Path
        EXAMPLES_DIR = os.path.dirname(DEFAULT_IMAGE_PATH)
        # EXAMPLES_DIR = "Data/Datasets/IM2LATEX_100K/IM2LATEX_100K/formula_images_processed/formula_images_processed/"
        EXAMPLE_FILES = os.listdir(EXAMPLES_DIR)
        USERINPUT_ImagePath = st.selectbox("Select Example File", EXAMPLE_FILES)
        USERINPUT_ImagePath = os.path.join(EXAMPLES_DIR, USERINPUT_ImagePath)
        USERINPUT_Image = open(USERINPUT_ImagePath, "rb").read()
    elif USERINPUT_LoadType == "Upload File":
        USERINPUT_Image = st.file_uploader("Upload Image", type=["jpg", "png", "PNG", "jpeg", "bmp"])
        if USERINPUT_Image is None: USERINPUT_Image = open(DEFAULT_IMAGE_PATH, "rb").read()
    elif USERINPUT_LoadType == "Datasets":
        # Select Dataset
        USERINPUT_Dataset = st.selectbox("Select Dataset", list(DATASETS.keys()))
        DATASET_MODULE = DATASETS[USERINPUT_Dataset]
        # Load Dataset
        DATASET = DATASET_MODULE.DATASET_FUNCS["test"]()
        N = DATASET.shape[0]
        # Display Top N Images
        N = DATASET.shape[0]
        USERINPUT_ViewSampleIndex = st.slider(f"View Sample ({N} Samples)", 0, N-1, 0, 1)
        st.image(DATASET["path"][USERINPUT_ViewSampleIndex], caption=f"Image: {USERINPUT_ViewSampleIndex}", use_column_width=True)
        st.latex(DATASET["code"][USERINPUT_ViewSampleIndex])
        USERINPUT_Image = open(DATASET["path"][USERINPUT_ViewSampleIndex], "rb").read()
    else:
        USERINPUT_Image = UI_GenerateLaTeXImage()
        temp_save_path = TEMP_PATH + "GeneratedImage.png"
        cv2.imwrite(temp_save_path, USERINPUT_Image)
        USERINPUT_Image = open(temp_save_path, "rb").read()

    # Show Image
    st.image(USERINPUT_Image, caption="Input Image", use_column_width=True)

    return USERINPUT_Image

def UI_ImageEdit(USERINPUT_Image):
    '''
    Clean and Edit Image
    '''
    st.markdown("## Image Edit")
    # Get Image Array
    I = ImageUtils_Bytes2Array(USERINPUT_Image)
    # Clean Image
    if st.checkbox("Clean Image"):
        I = ImageUtils_Clean(I)
        st.image(I, caption=f"Cleaned Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Sharpen
    if st.checkbox("Sharpen Image"):
        I = ImageUtils_Effect_Sharpen(I)
        I = ImageUtils_Effect_Normalise(I)
        st.image(I, caption=f"Sharpened Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Binarise
    if st.checkbox("Binarise Image"):
        USERINPUT_BinariseThreshold = st.slider("Binarise Threshold 1", 0.0, 1.0, 0.1, 0.01)
        I = ImageUtils_Effect_Binarise(I, threshold=USERINPUT_BinariseThreshold)
        st.image(I, caption=f"Binarised Image 1 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Resize
    if st.checkbox("Erode Image"):
        USERINPUT_ResizeMaxSize = st.number_input("Resize Max Size 1", 1, 2048, 1024, 1)
        I = ImageUtils_Resize(I, maxSize=USERINPUT_ResizeMaxSize)
        st.image(I, caption=f"Resized Image 1 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
        # Erode
        USERINPUT_Iterations = st.slider("Erosion Iterations", 0, 10, 1, 1)
        I = ImageUtils_Effect_Erode(I, iterations=USERINPUT_Iterations)
        # I = ImageUtils_Effect_Normalise(I)
        st.image(I, caption=f"Eroded Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
        # Resize
        USERINPUT_ResizeMaxSize = st.number_input("Resize Max Size 2", 1, 2048, 1024, 1)
        I = ImageUtils_Resize(I, maxSize=USERINPUT_ResizeMaxSize)
        st.image(I, caption=f"Resized Image 2 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
        # Binarise
        USERINPUT_BinariseThreshold = st.slider("Binarise Threshold 2", 0.0, 1.0, 0.1, 0.01)
        I = ImageUtils_Effect_Binarise(I, threshold=USERINPUT_BinariseThreshold)
        st.image(I, caption=f"Binarised Image 2 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Normalise
    if st.checkbox("Normalise Image"):
        I = ImageUtils_Effect_Normalise(I)
        st.image(I, caption=f"Normalised Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Partition
    bg_white = (1.0 - I.mean()) < (I.mean() - 0.0)
    if st.checkbox("Partition Image"):
        PartitionData = ImageUtils_Partition_Horizontal(I, threshold=0.99 if bg_white else 0.01, bg_white=bg_white, display=False)
        USERINPUT_ShowCount = st.number_input("Show Partitions Count", 1, len(PartitionData["partitions"]), 1, 1)
        SelectedPartitions = PartitionData["partitions"][:USERINPUT_ShowCount]
        I = ImageUtils_PartitionUtils_PartImage(I, SelectedPartitions, bg_white=bg_white)
    # Padding
    if st.checkbox("Pad Image"):
        cols = st.columns(4)
        PadSizes = [0, 0, 0, 0]
        PadNames = ["Top", "Bottom", "Left", "Right"]
        PadValue = 1.0 if bg_white else 0.0
        for i in range(len(PadNames)):
            PadSizes[i] = cols[i].number_input(f"Pad Size: {PadNames[i]}", 0, 2048, 5, 1)
        I = ImageUtils_Pad(I, PadSizes, PadValue)
        st.image(I, caption=f"Padded Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Resize
    if st.checkbox("Resize Image"):
        USERINPUT_ResizeMaxSize = st.number_input("Resize Max Size", 1, 2048, 256, 1)
        I = ImageUtils_Resize(I, maxSize=USERINPUT_ResizeMaxSize)
        st.image(I, caption=f"Resized Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Flip
    USERINPUT_InvertColor = st.checkbox("Invert Image Color")
    if USERINPUT_InvertColor: I = ImageUtils_Effect_InvertColor(I)

    st.markdown("## Final Image")
    print("Input:", I.shape, I.dtype, I.min(), I.max())
    I_final = np.array(I * 255.0, dtype=np.uint8)
    st.image(I_final, caption=f"Final Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # HIST_PLOT = ImageUtils_PlotImageHistogram(I_final)
    # st.plotly_chart(HIST_PLOT)
    # Convert to Bytes
    TempImagePath = TEMP_PATH + "CleanedImage.png"
    ImageUtils_SaveImage(I_final, TempImagePath)
    USERINPUT_Image = open(TempImagePath, "rb").read()

    return USERINPUT_Image

def UI_DisplayLatexEquations(latex_code):
    '''
    Display List of Latex Equations
    '''
    for i, eq in enumerate(latex_code):
        st.markdown("### Equation {}".format(i+1))
        st.markdown("```\n" + eq + "\n```")
        st.latex(eq)

def UI_DisplayCombinedLatexEquations(latex_code, ref_code):
    '''
    Display List of Latex Equations
    '''
    for i, eq in enumerate(latex_code):
        st.markdown("### Equation {}".format(i+1))
        st.markdown("```\n" + ref_code[i] + "\n" + eq + "\n```")
        st.latex(eq)

def UI_DisplayTestResults(TestData):
    '''
    Display Test Results
    '''
    # Test Results
    st.markdown("## Test Results")
    y_true, y_pred = np.array(TestData["y_true"]), np.array(TestData["y_pred"])
    n_matches = np.count_nonzero(y_true == y_pred)
    Results = {
        "Num Matches": n_matches,
        "Accuracy": n_matches / y_true.shape[0]
    }
    st.write(Results)
    # Test Times
    st.markdown("## Test Times")
    Times = {
        "Min Time": float(np.min(TestData["time_exec"])),
        "Max Time": float(np.max(TestData["time_exec"])),
        "Mean Time": float(np.mean(TestData["time_exec"])),
        "Median Time": float(np.median(TestData["time_exec"])),
        "Std Time": float(np.std(TestData["time_exec"]))
    }
    st.write(Times)
    # Test Images
    st.markdown("## Test Images")
    ImageSizes = {
        "height": {
            "Min": float(np.min(TestData["image_size"]["height"])),
            "Max": float(np.max(TestData["image_size"]["height"])),
            "Mean": float(np.mean(TestData["image_size"]["height"])),
            "Median": float(np.median(TestData["image_size"]["height"])),
            "Std": float(np.std(TestData["image_size"]["height"]))
        },
        "width": {
            "Min": float(np.min(TestData["image_size"]["width"])),
            "Max": float(np.max(TestData["image_size"]["width"])),
            "Mean": float(np.mean(TestData["image_size"]["width"])),
            "Median": float(np.median(TestData["image_size"]["width"])),
            "Std": float(np.std(TestData["image_size"]["width"]))
        }
    }
    st.write(ImageSizes)

# Main Functions
def formula_ocr_single_image():
    # Title
    st.markdown("# Formula OCR - Single Image")

    # Load Inputs
    # Method
    USERINPUT_OCRMethod = st.sidebar.selectbox(
        "Select OCR Method",
        list(OCR_MODULES.keys())
    )
    if USERINPUT_OCRMethod == "FormulaGAN": UI_LoadModel_FormulaGAN()
    USERINPUT_OCRMethod = OCR_MODULES[USERINPUT_OCRMethod]
    # Image
    USERINPUT_Image = UI_LoadImage()
    # Clean and Edit Image
    USERINPUT_Image = UI_ImageEdit(USERINPUT_Image)

    # Process Inputs
    if st.button("Run OCR"):
        # OCR
        LATEX_CODE = USERINPUT_OCRMethod["convert_image_to_latex"](USERINPUT_Image)

        # Display Outputs
        st.markdown("## Latex Output")
        UI_DisplayLatexEquations(LATEX_CODE)
    print()

def formula_ocr_single_image_combined():
    # Title
    st.markdown("# Combined Formula OCR - Single Image")

    # Load Inputs
    # Methods
    OCR_ref = OCR_MODULES["Pix2Tex"]
    OCR_gan = OCR_MODULES["FormulaGAN"]
    UI_LoadModel_FormulaGAN()
    # Image
    USERINPUT_Image = UI_LoadImage()
    # Clean and Edit Image
    USERINPUT_Image = UI_ImageEdit(USERINPUT_Image)

    # Process Inputs
    if st.button("Run OCR"):
        # Run Reference OCR
        LATEX_CODE_REF = OCR_ref["convert_image_to_latex"](USERINPUT_Image)[0]
        LATEX_CODE_REF = LATEX_CODE_REF.replace(" ", "")
        TOKENS_REF = LaTeXParse_Tokenize(LATEX_CODE_REF, tokens=list(OCR_gan["tokenizer"].get_vocabulary()))
        LATEX_CODE_REF = " ".join(TOKENS_REF)

        # Use Tokens from Reference OCR to Run GAN OCR
        TOKENS_PRED = []
        progressObj = st.progress(0)
        for i in range(len(TOKENS_REF)):
            cur_partial_code = " ".join(TOKENS_REF[:i])
            LATEX_CODE_GAN = OCR_gan["convert_image_to_latex"](USERINPUT_Image, partial_code=cur_partial_code)[0]
            pred_tokens = LATEX_CODE_GAN.split(" ")
            if i < len(pred_tokens): TOKENS_PRED.append(pred_tokens)#TOKENS_PRED.append(pred_tokens[i])
            progressObj.progress((i+1) / len(TOKENS_REF))
        LATEX_CODES_GAN = [" ".join(t) for t in TOKENS_PRED]
        # Display Outputs
        st.markdown("## Latex Output - Ref")
        UI_DisplayLatexEquations([LATEX_CODE_REF])
        st.markdown("## Latex Output - GAN - Stepwise")
        LATEX_CODES_GAN_REF = [" ".join(TOKENS_REF[:i]) for i in range(len(LATEX_CODES_GAN))]
        UI_DisplayCombinedLatexEquations(LATEX_CODES_GAN, LATEX_CODES_GAN_REF)

def formula_ocr_dataset_test():
    # Title
    st.markdown("# Formula OCR - Dataset Test")

    # Load Inputs
    # Method
    USERINPUT_OCRMethod = st.sidebar.selectbox(
        "Select OCR Method",
        list(OCR_MODULES.keys())
    )
    if USERINPUT_OCRMethod == "FormulaGAN": UI_LoadModel_FormulaGAN()
    USERINPUT_OCRMethod = OCR_MODULES[USERINPUT_OCRMethod]
    # Dataset
    USERINPUT_Dataset = UI_LoadDataset()

    USERINPUT_DisplayLatestProcessedSample = st.checkbox("Display Latest Processed Sample", value=True)
    # Process Inputs
    if st.button("Run Dataset Test"):
        TestData = {
            "y_true": [],
            "y_pred": [],
            "time_exec": [],
            "image_size": {
                "height": [],
                "width": []
            }
        }
        PROGRESS_OBJ = st.progress(0.0)
        CURRENT_RESULT_OBJS = {
            "image": st.empty(),
            "true": {
                "title": st.empty(),
                "latex": st.empty(),
                "code": st.empty()
            },
            "pred": {
                "title": st.empty(),
                "latex": st.empty(),
                "code": st.empty()
            }
        }
        for i in range(USERINPUT_Dataset.shape[0]):
            I_path = USERINPUT_Dataset["path"][i]
            LATEX_CODE_TRUE = USERINPUT_Dataset["code"][i]
            # Read Image
            I_bytes = open(I_path, "rb").read()
            I_shape = np.array(cv2.imread(I_path)).shape[:2]
            TestData["image_size"]["height"].append(I_shape[0])
            TestData["image_size"]["width"].append(I_shape[1])
            # OCR
            START_TIME = time.time()
            LATEX_CODE_PRED = USERINPUT_OCRMethod["convert_image_to_latex"](I_bytes)
            END_TIME = time.time()
            LATEX_CODE_PRED = LATEX_CODE_PRED[0]
            # Update Progress
            TestData["y_true"].append(LATEX_CODE_TRUE)
            TestData["y_pred"].append(LATEX_CODE_PRED)
            TestData["time_exec"].append(END_TIME - START_TIME)
            PROGRESS_OBJ.progress((i+1) / USERINPUT_Dataset.shape[0])
            if USERINPUT_DisplayLatestProcessedSample:
                CURRENT_RESULT_OBJS["image"].image(I_path, caption=f"Image: {i}", use_column_width=True)
                CURRENT_RESULT_OBJS["true"]["title"].markdown("True:")
                CURRENT_RESULT_OBJS["true"]["latex"].latex(LATEX_CODE_TRUE)
                CURRENT_RESULT_OBJS["true"]["code"].markdown("```\n" + LATEX_CODE_TRUE + "\n```")
                CURRENT_RESULT_OBJS["pred"]["title"].markdown("Predicted:")
                CURRENT_RESULT_OBJS["pred"]["latex"].latex(LATEX_CODE_PRED)
                CURRENT_RESULT_OBJS["pred"]["code"].markdown("```\n" + LATEX_CODE_PRED + "\n```")
        # Display Outputs
        UI_DisplayTestResults(TestData)

def image_process():
    # Title
    st.markdown("# Image Process")

    # Image
    USERINPUT_Image = UI_LoadImage()
    # Clean and Edit Image
    USERINPUT_Image = UI_ImageEdit(USERINPUT_Image)

    # Partition Image
    st.markdown("## Partition Image")
    I = ImageUtils_Bytes2Array(USERINPUT_Image)
    PartitionData = ImageUtils_Partition_Horizontal(I, threshold=0.99, bg_white=True, display=False)
    I_partitioned = ImageUtils_PartitionUtils_DisplayPartitions(I, PartitionData["partitions"], thickness=1)
    st.plotly_chart(PartitionData["figs"]["counts"])
    st.pyplot(PartitionData["figs"]["combined"])
    st.image(I_partitioned, caption="Partitioned Image", use_column_width=True)

# Mode Vars
MODES = {
    "Formula OCR - Single Image": formula_ocr_single_image,
    "Formula OCR - Single Image - Combined": formula_ocr_single_image_combined,
    "Formula OCR - Dataset Test": formula_ocr_dataset_test,
    "Image Process": image_process
}

# App Functions
def app_main():
    # Title
    st.markdown("# Formula GAN Project")
    # Mode
    USERINPUT_Mode = st.sidebar.selectbox(
        "Select Mode",
        list(MODES.keys())
    )
    MODES[USERINPUT_Mode]()

# RunCode
if __name__ == "__main__":
    if not os.path.isdir(TEMP_PATH):
        os.makedirs(TEMP_PATH)
    # Assign Objects
    
    # Run Main
    app_main()

# Good Examples for IM2LATEX Dataset:
# 7785, 