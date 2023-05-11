"""
Process Images through Pretrained Models
"""

# Imports
import tensorflow.keras as TFKeras
from keras.preprocessing import image

from .Utils import *

# Main Vars
PRETRAINED_MODELS = {
    "ResNet50": TFKeras.applications.ResNet50,
    "Xception": TFKeras.applications.Xception,
    "InceptionV3": TFKeras.applications.InceptionV3,
    "InceptionResNetV2": TFKeras.applications.InceptionResNetV2,
    "VGG19": TFKeras.applications.VGG19
}

# Main Functions
# Process Functions
def Model_PretrainedCNN_Process(
    I_paths, save_dir, 
    model="ResNet50", model_target_size=(224, 224),
    **params
    ):
    '''
    Pretrained CNN - Process Images
        - Input: Image Paths
        - Output: Feature Vectors
    '''
    # Get Pretrained Model
    model_pretrained = PRETRAINED_MODELS[model](
        weights="imagenet",
        include_top=False
    )
    
    # Process
    for path in tqdm(I_paths):
        # Set Path
        fname = os.path.splitext(os.path.basename(path))[0]
        savepath = os.path.join(save_dir, fname + ".npy")
        # Load Image
        I = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(I)
        x = np.expand_dims(x, axis=0)
        x = PRETRAINED_MODELS[model].preprocess_input(x)
        # Get Features
        x_features = model_pretrained.predict(x)
        x_features = np.reshape(x_features , x_features.shape[1])
        # Save
        np.save(savepath, x_features)