# FormulaGAN

# Requirements
Install required python modules from [requirements.txt](requirements.txt).

# Dataset Setup
 - Download IM2LATEX_100K Dataset from any one source
   - Kaggle Link: [https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k?resource=download](https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k?resource=download)
   - Drive Link: [https://drive.google.com/file/d/1UcRILO4BYcFGDlqOZz5h7FXovz5dNtgT/view?usp=sharing](https://drive.google.com/file/d/1UcRILO4BYcFGDlqOZz5h7FXovz5dNtgT/view?usp=sharing)
 - Unzip the dataset
   - Copy **ONLY** the **formula_images_processed/** folder to [Data/Datasets/IM2LATEX_100K/IM2LATEX_100K/formula_images_processed/](Data/Datasets/IM2LATEX_100K/IM2LATEX_100K/formula_images_processed/)
   - Do not copy the csv files as edited csv files are given in the repository.
 - To check if the dataset is loaded, run [app_Dataset.py](app_Dataset.py) app and visualise the dataset.

# Models Setup
 - Download the models
   - Drive Link: [https://drive.google.com/drive/folders/1NT9H8gPNxsWbinG7dpmQjE8ZIHkgZoo9?usp=sharing](https://drive.google.com/drive/folders/1NT9H8gPNxsWbinG7dpmQjE8ZIHkgZoo9?usp=sharing)
 - Place the .h5 files in the [Models/FormulaGAN/](Models/FormulaGAN) folder.

# Apps
## Main
 - [app.py](app.py)
 - Shell command to run, 
    ```shell 
    streamlit run app.py
    ```
 - Formula OCR - Single Image: Test Pix2Tex and FormulaGAN model on an image
 - Formula OCR - Single Image - Combined: Test Combined method on an image
 - Formula OCR - Dataset Test: Test model using dataset

## Model
 - [app_Model.py](app_Model.py)
 - Shell command to run, 
    ```shell 
    streamlit run app_Model.py
    ```
 - Model Visualise: Visualise a keras model
 - Download and **install** Graphviz( https://graphviz.gitlab.io/download/) and put it on PATH for the [app_Model.py](app_Model.py) visualisers to work

## Dataset
 - [app_Dataset.py](app_Dataset.py)
 - Shell command to run, 
    ```shell 
    streamlit run app_Dataset.py
    ```
 - Generate LaTeX Image
 - Generate and Visualise Dataset

## Train
 - [app_Train.py](app_Train.py)
 - Shell command to run, 
    ```shell 
    streamlit run app_Train.py
    ```
 - FormulaGAN Train: Encoder-Decoder: Train Encoder-Decoder model
 - FormulaGAN Train: GAN: Train GAN model