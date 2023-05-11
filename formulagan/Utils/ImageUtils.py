"""
Image Utils
"""

# Imports
import io
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Main Functions
# Conversion Functions
def ImageUtils_Bytes2Array(I_bytes):
    '''
    Image Bytes to Array
    '''
    # Load Image
    I_array = np.array(Image.open(io.BytesIO(I_bytes)), dtype=float)
    # print("Utils:", I_array.shape, I_array.dtype, I_array.min(), I_array.max())
    # Greyscale
    if I_array.ndim == 3: I_array = np.mean(I_array[:, :, :3], axis=-1)
    # Normalize
    if I_array.max() > 1.0: I_array /= 255.0

    return I_array

def ImageUtils_Array2Bytes(I):
    '''
    Array to Bytes
    '''
    I = np.array(I * 255.0, dtype=np.uint8)
    I_bytes = I.tobytes()

    return I_bytes

def ImageUtils_SaveImage(I, filename):
    '''
    Save Image
    '''
    # Save
    cv2.imwrite(filename, I)

# Plot Functions
def ImageUtils_PlotImageHistogram(I, bins=256):
    '''
    Plot Image Histogram
    '''
    # Convert Values
    vals = I.ravel()
    # Plot Histogram
    fig = plt.figure()
    plt.hist(vals, bins=bins, range=(I.min(), I.max()))
    plt.title("Image Histogram")

    return fig

# Clean Functions
def ImageUtils_Clean(I):
    '''
    Clean Image
     - Normalise
     - Change Background to Black
    '''
    # Normalise
    I_normalised = (I - I.min()) / (I.max() - I.min())
    # Flip Background Color to Black Always
    # If I_mean is closer to I_max than I_min, then flip as background is currently I_max, i.e. white
    flip = (1.0 - I_normalised.mean()) < (I_normalised.mean() - 0.0)
    if flip: I_normalised = 1.0 - I_normalised

    I_cleaned = I_normalised

    return I_cleaned

def ImageUtils_Resize(I, maxSize=1024):
    '''
    Resize Image to have max width or height as given
    '''
    aspect_ratio = I.shape[1] / I.shape[0]
    newSize = (maxSize, maxSize)
    if aspect_ratio > 1:
        newSize = (int(maxSize / aspect_ratio), maxSize)
    else:
        newSize = (maxSize, int(maxSize * aspect_ratio))
    newSize = newSize[::-1]

    I_resized = cv2.resize(I, newSize)

    return I_resized

# Padding Functions
def ImageUtils_Pad(I, padSizes=[0, 0, 0, 0], padValue=0.0):
    '''
    Pad Image: Top, Bottom, Left, Right
    '''
    # Pad
    I_FinalShape = (I.shape[0] + padSizes[0] + padSizes[1], I.shape[1] + padSizes[2] + padSizes[3])
    I_padded = np.ones(I_FinalShape, dtype=I.dtype) * padValue
    I_padded[padSizes[0]:padSizes[0]+I.shape[0], padSizes[2]:padSizes[2]+I.shape[1]] = I

    return I_padded

# Effect Functions
def ImageUtils_Effect_InvertColor(I):
    '''
    Invert Image Values
    '''
    I_flipped = 1.0 - I

    return I_flipped

def ImageUtils_Effect_Normalise(I):
    '''
    Normalise Image
    '''
    I_normalised = (I - I.min()) / (I.max() - I.min())

    return I_normalised

def ImageUtils_Effect_Binarise(I, threshold=0.5):
    '''
    Binarise Image
    '''
    I_binarised = I > threshold
    I_binarised = np.array(I_binarised, dtype=float)

    return I_binarised

def ImageUtils_Effect_Sharpen(I):
    '''
    Sharpen Image
    '''
    # Sharpen
    KERNEL_SHARPEN_1 = np.array([
        [0., -1/5, 0.],
        [-1/5, 1, -1/5],
        [0., -1/5, 0.]
    ])
    I_shapened = cv2.filter2D(src=I, ddepth=-1, kernel=KERNEL_SHARPEN_1)
    # Clip
    I_shapened = np.clip(I_shapened, 0.0, 1.0)

    return I_shapened

def ImageUtils_Effect_Erode(I, iterations=1):
    '''
    Erode Image
    '''
    # Erode
    KERNEL_ERODE_1 = np.ones((3, 3))
    I_eroded = cv2.erode(I, KERNEL_ERODE_1, iterations=iterations)

    return I_eroded

# Image Partition Functions
def ImageUtils_Partition_Horizontal(I, threshold=0.5, bg_white=True, display=False):
    '''
    Partition Image Horizontally between charecters
    '''
    # Active Pixels Count - Number of active pixels in each vertical column of image
    # Threshold Image to form binary image
    I_binarised = ImageUtils_Effect_Binarise(I, threshold=threshold)
    # Find Active Pixels Counts
    if bg_white: I_binarised = 1.0 - I_binarised
    ActivePixelCounts = np.sum(I_binarised, axis=0)

    # Display
    FIG_COUNTS = plt.figure()
    plt.plot(ActivePixelCounts)
    plt.title("Active Pixel Counts")
    if display: plt.show()
    FIG_COMBINED = plt.figure()
    plt.imshow(I_binarised, cmap='gray')
    plt.plot(ActivePixelCounts)
    plt.title("Active Pixel Counts - Image with Counts")
    if display: plt.show()

    # Partition
    PartitionPoints = []
    curPart = -1
    for i in range(len(ActivePixelCounts)):
        # If not in a partition and there are active pixels, start a new partition
        if ActivePixelCounts[i] > 0:
            if curPart == -1:
                curPart = i
        # If in a partition and there are no active pixels, end the partition
        elif curPart != -1:
            PartitionPoints.append((curPart, i-1))
            curPart = -1
    # If in a partition at the end of the image, end the partition
    if curPart != -1:
        PartitionPoints.append((curPart, len(ActivePixelCounts)-1))

    out = {
        "figs": {
            "counts": FIG_COUNTS, 
            "combined": FIG_COMBINED
        },
        "active_pixel_counts": ActivePixelCounts,
        "partitions": PartitionPoints
    }
    return out

def ImageUtils_PartitionUtils_DisplayPartitions(I, partitions, thickness=1):
    '''
    Partition Image into Boxes
    '''
    # Fix Image
    I = np.array(I * 255.0, dtype=np.uint8)
    I = np.dstack((I, I, I))
    # Form Display Image
    PartitionBoxes = []
    for (start, end) in partitions:
        I = cv2.rectangle(I, (start, 0), (end, I.shape[0]-1), (0, 255, 0), thickness)

    return I

def ImageUtils_PartitionUtils_PartImage(I, partitions, bg_white=True):
    '''
    Get Partitioned Image from Reference Image
    '''
    # Init Partitioned Image
    I_part = np.ones(I.shape) * (1.0 if bg_white else 0.0)
    # Apply Partitions from Reference Image
    for (start, end) in partitions:
        I_part[:, start:end+1] = I[:, start:end+1]

    return I_part