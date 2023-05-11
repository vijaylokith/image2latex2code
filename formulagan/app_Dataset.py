"""
Streamlit App - Dataset
"""

# Imports
import os
import cv2
import streamlit as st
from tqdm import tqdm

from FormulaOCR import *
from Utils.ImageUtils import *

# Main Vars
DEFAULT_CODES_PATH = "Data/InputCodes/Test.txt"
SAVE_DATASET_PATH = "Data/Datasets/GeneratedDatasets/Test/"
TEMP_PATH = "Data/Temp/"

# Dataset Threshold Functions
def DatasetDF_ThresholdCodeLength(DATASET_df, max_sequence_length=200):
    '''
    Threshold Code Size and drop rows with large lengths
    '''
    Codes = np.array(DATASET_df["code"], dtype=str)
    CodeLengths = np.array([len(code.split(" ")) for code in Codes])
    # CodeLengths = DATASET_df["code"].str.split(" ").apply(len)
    DATASET_df = DATASET_df.drop(DATASET_df[CodeLengths > max_sequence_length].index)
    DATASET_df.reset_index(drop=True, inplace=True)

    return DATASET_df

# Plot Functions
def Plot_UniqueValCounts(data, title=""):
    '''
    Plot the Unique value counts as bar chart
    '''
    fig = plt.figure()
    data_unique, data_unique_counts = np.unique(data, return_counts=True)
    sorted_indices = np.argsort(data_unique)
    plt.bar(data_unique[sorted_indices], data_unique_counts[sorted_indices])
    plt.title(title)

    return fig

# UI Functions
def UI_LoadCodes():
    '''
    Load Image
    '''
    # Load Image
    st.markdown("## Load Codes")
    USERINPUT_LoadType = st.selectbox("Load Type", ["Examples", "Upload File"])
    if USERINPUT_LoadType == "Examples":
        # Load Filenames from Default Path
        EXAMPLES_DIR = os.path.dirname(DEFAULT_CODES_PATH)
        EXAMPLE_FILES = os.listdir(EXAMPLES_DIR)
        USERINPUT_CodesPath = st.selectbox("Select Example File", EXAMPLE_FILES)
        USERINPUT_CodesPath = os.path.join(EXAMPLES_DIR, USERINPUT_CodesPath)
        USERINPUT_Codes = open(USERINPUT_CodesPath, "r").read().split("\n")
    else:
        USERINPUT_Codes = st.file_uploader("Upload Codes File", type=["txt"])
        if USERINPUT_Codes is None: USERINPUT_Codes = open(DEFAULT_CODES_PATH, "r").read().split("\n")
        else: USERINPUT_Codes = USERINPUT_Codes.split("\n")
    # Display
    st.markdown("Found **" + str(len(USERINPUT_Codes)) + "** Codes!")

    return USERINPUT_Codes

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

def UI_LoadDataset():
    '''
    Load Dataset
    '''
    st.markdown("## Load Dataset")
    # Select Dataset
    USERINPUT_Dataset = st.selectbox("Select Dataset", list(DATASETS.keys()))
    DATASET_MODULE = DATASETS[USERINPUT_Dataset]
    # Load Dataset
    DATASET = DATASET_MODULE.DATASET_FUNCS["full"]()
    # Remove NaN
    DATASET.dropna(inplace=True)
    DATASET.reset_index(drop=True, inplace=True)

    # Threshold Dataset
    USERINPUT_ThresholdDataset = st.checkbox("Threshold Dataset")
    if USERINPUT_ThresholdDataset:
        USERINPUT_MaxCodeLength = st.number_input("Max Code Length", 1, 1000, 200, 1)
        DATASET = DatasetDF_ThresholdCodeLength(DATASET, max_sequence_length=USERINPUT_MaxCodeLength)

    # Display Top N Images
    N = DATASET.shape[0]
    USERINPUT_ViewSampleIndex = st.slider(f"View Sample ({N} Samples)", 0, N-1, 0, 1)
    I = np.array(cv2.imread(DATASET["path"][USERINPUT_ViewSampleIndex]), dtype=np.uint8)
    st.image(
        DATASET["path"][USERINPUT_ViewSampleIndex], 
        caption=f"Image: {USERINPUT_ViewSampleIndex} {I.shape}", 
        use_column_width=True
    )
    st.latex(DATASET["code"][USERINPUT_ViewSampleIndex])

    return DATASET

def UI_VisualiseDatasetImages(DATASET):
    '''
    Visualisations on Dataset Images
    '''
    # Load Images Info
    IMAGES_SIZES = []
    N = DATASET.shape[0]
    progressObj = st.progress(0.0)
    for i in tqdm(range(N)):
        I = np.array(cv2.imread(DATASET["path"][i]))
        IMAGES_SIZES.append((I.shape[0], I.shape[1]))
        progressObj.progress((i+1) / N)
    IMAGES_SIZES = np.array(IMAGES_SIZES)
    ASPECT_RATIOS = IMAGES_SIZES[:, 0] / IMAGES_SIZES[:, 1]
    # Images Plot Visualisations
    HEIGHTS_FIG = Plot_UniqueValCounts(IMAGES_SIZES[:, 0], title="Heights")
    WIDTHS_FIG = Plot_UniqueValCounts(IMAGES_SIZES[:, 1], title="Widths")
    ASPECT_RATIOS_FIG = plt.figure()
    plt.hist(ASPECT_RATIOS, bins=100)
    plt.title("Aspect Ratios")
    # Display
    IMAGES_VIS = {
        "Num Images": N,
        "Height": {
            "Min": int(np.min(IMAGES_SIZES[:, 0])),
            "Max": int(np.max(IMAGES_SIZES[:, 0])),
            "Mean": np.mean(IMAGES_SIZES[:, 0]),
            "Median": np.median(IMAGES_SIZES[:, 0]),
            "Std": np.std(IMAGES_SIZES[:, 0])
        },
        "Width": {
            "Min": int(np.min(IMAGES_SIZES[:, 1])),
            "Max": int(np.max(IMAGES_SIZES[:, 1])),
            "Mean": np.mean(IMAGES_SIZES[:, 1]),
            "Median": np.median(IMAGES_SIZES[:, 1]),
            "Std": np.std(IMAGES_SIZES[:, 1])
        },
        "Aspect Ratio": {
            "Min": np.min(ASPECT_RATIOS),
            "Max": np.max(ASPECT_RATIOS),
            "Mean": np.mean(ASPECT_RATIOS),
            "Median": np.median(ASPECT_RATIOS),
            "Std": np.std(ASPECT_RATIOS)
        }
    }
    st.markdown("### Images")
    st.write(IMAGES_VIS)
    st.plotly_chart(HEIGHTS_FIG)
    st.plotly_chart(WIDTHS_FIG)
    st.plotly_chart(ASPECT_RATIOS_FIG)

def UI_VisualiseDataset(DATASET):
    '''
    Standard Visualisations on Dataset
    '''
    st.markdown("## Visualisations")

    # Count Visualisations
    CODES = np.array(DATASET["code"], dtype=str)
    UNIQUE_CODES, UNIQUE_CODES_COUNTS = np.unique(CODES, return_counts=True)
    INDICES_TOPN = np.argsort(UNIQUE_CODES_COUNTS)[::-1][:min(UNIQUE_CODES_COUNTS.shape[0], 3)]
    TOPN_CODES = [{"count": int(UNIQUE_CODES_COUNTS[i]), "code": UNIQUE_CODES[i]} for i in INDICES_TOPN]
    COUNT_VIS = {
        "Num Samples": CODES.shape[0],
        "Num Unique Codes": UNIQUE_CODES.shape[0],
        "Top 3 Unique Codes": TOPN_CODES,
    }
    # Count Plot Visualisations
    COUNT_FIG = Plot_UniqueValCounts(UNIQUE_CODES_COUNTS, title="Unique Codes Counts")
    # Display
    st.markdown("### Counts")
    st.write(COUNT_VIS)
    st.plotly_chart(COUNT_FIG)
    
    # Code Visualisations
    UNIQUE_CODES_TOKENS = [code.split(" ") for code in UNIQUE_CODES]
    # Find Code Lengths, tokens separated by " "
    UNIQUE_CODES_LENGTHS = np.array([len(tokens) for tokens in UNIQUE_CODES_TOKENS])
    CODE_VIS = {
        "Minimum Code Length": int(np.min(UNIQUE_CODES_LENGTHS)),
        "Maximum Code Length": int(np.max(UNIQUE_CODES_LENGTHS)),
        "Mean Code Length": np.mean(UNIQUE_CODES_LENGTHS),
        "Median Code Length": np.median(UNIQUE_CODES_LENGTHS),
        "Standard Deviation Code Length": np.std(UNIQUE_CODES_LENGTHS)
    }
    # Code Plot Visualisations
    CODELENGTH_FIG = Plot_UniqueValCounts(UNIQUE_CODES_LENGTHS, title="Code Lengths")
    # Display
    st.markdown("### Code Lengths")
    st.write(CODE_VIS)
    st.plotly_chart(CODELENGTH_FIG)
    # Find Code Vocab
    UNIQUE_CODES_VOCAB = []
    vocab = [set(tokens) for tokens in UNIQUE_CODES_TOKENS]
    for tokens in vocab: UNIQUE_CODES_VOCAB.extend(tokens)
    UNIQUE_CODES_VOCAB = np.unique(UNIQUE_CODES_VOCAB)
    CODE_VOCAB_VIS = {
        "Vocab Size": UNIQUE_CODES_VOCAB.shape[0]
    }
    # Display
    st.markdown("### Code Vocab")
    st.write(CODE_VOCAB_VIS)

    # Images Visualisations
    if st.button("Visualise Images"):
        UI_VisualiseDatasetImages(DATASET)

# Mode Functions
def generate_latex_image():
    # Title
    st.markdown("# Generate LaTeX Image")

    # Params
    USERINPUT_Code = UI_LoadLaTeXCode()
    USERINPUT_Euler = st.checkbox("Euler")
    USERINPUT_DPI = st.number_input("DPI", 1, 5000, 500, 1)

    # Convert to Image
    USERINPUT_Image = DATASETS["GeneratedDatasets"].DatasetUtils_LaTeX2Image(USERINPUT_Code, euler=USERINPUT_Euler, dpi=USERINPUT_DPI)

    # Display
    st.markdown("## Outputs")
    st.image(USERINPUT_Image, caption=f"LaTeX Image {USERINPUT_Image.shape[:2]}", use_column_width=True)

def generate_dataset():
    # Title
    st.markdown("# Generate LaTeX Dataset")

    # Params
    USERINPUT_Codes = UI_LoadCodes()
    USERINPUT_Euler = st.checkbox("Euler")
    USERINPUT_DPI = st.number_input("DPI", 1, 5000, 500, 1)
    USERINPUT_SavePath = st.text_input("Save Path", SAVE_DATASET_PATH)

    # Generate Dataset
    if st.button("Generate Dataset"):
        DATASETS["GeneratedDatasets"].DatasetUtils_GenerateLaTeXImageDataset(
            USERINPUT_Codes, USERINPUT_SavePath, 
            euler=USERINPUT_Euler, dpi=USERINPUT_DPI,
            progressObj=st.progress(0.0)
        )

def visualise_dataset():
    # Title
    st.markdown("# Visualise Dataset")

    # Load Inputs
    USERINPUT_Dataset = UI_LoadDataset()

    # Visualise Dataset
    UI_VisualiseDataset(USERINPUT_Dataset)

# Main Vars
MODES = {
    "Generate LaTeX Image": generate_latex_image,
    "Generate Dataset": generate_dataset,
    "Visualise Dataset": visualise_dataset
}

# Main Functions
def app_main():
    # Title
    st.markdown("# LaTeX Dataset Utils")

    # Load Inputs
    # Method
    USERINPUT_Mode = st.sidebar.selectbox("Select Mode", list(MODES.keys()))
    USERINPUT_ModeFunc = MODES[USERINPUT_Mode]
    USERINPUT_ModeFunc()


# RunCode
if __name__ == "__main__":
    # Assign Objects
    
    # Run Main
    app_main()