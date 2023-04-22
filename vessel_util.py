import cv2 
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
from skimage.filters import sato
import numpy as np

DATA_DIR = "data"
GOLD_STANDARD_DIR = "manual1" 
FOV_DIR = "mask" 
IMAGE_DIR = "images"

def load_image(sample_name: str):
    sample_dir = Path(DATA_DIR) / Path(IMAGE_DIR)

    for ext in [ '.jpg', '.JPG' ]:
        sample = sample_dir / Path(sample_name + ext)
        if sample.is_file: 
            return cv2.imread(str(sample))

    raise FileNotFoundError

def load_fov(sample_name: str):
    sample_dir = Path(DATA_DIR) / Path(FOV_DIR)

    sample = sample_dir / Path(sample_name + '_mask.tif')
    if not sample.is_file:
        raise FileNotFoundError

    return cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)

def load_gold_standard(sample_name: str):
    sample_dir = Path(DATA_DIR) / Path(GOLD_STANDARD_DIR)

    sample = sample_dir / Path(sample_name + '.tif')
    if not sample.is_file:
        raise FileNotFoundError

    # load image in grayscale
    return cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)

def get_all_sample_names() -> List[str]:
    """ Return list of strings - names of given samples. We can get from names to images, fovs, and gold_standards """
    
    # Use files from Gold Standards, because they have basic names + tif extension
    samples_dir = Path(DATA_DIR) / Path(GOLD_STANDARD_DIR)

    # data/manual1/01_dr.tif -> 01_dr
    sample_names = [ sample_file.stem for sample_file in samples_dir.iterdir() ] 
    return sample_names

def plt_compare_two_images(image1, image2, title1 = None, title2 = None, grayscale = False):
    cmap = 'gray' if grayscale else 'viridis'
    plt.subplot(121)
    if title1 is not None: 
        plt.title(title1)
    plt.imshow(image1, cmap=cmap)
    plt.subplot(122)
    if title2 is not None:
        plt.title(title2)
    plt.imshow(image2, cmap=cmap)
    
# --- PROCESSING --- 

# TODO: use Fov?
def process_image_using_cv(image, fov, threshold: int, channel_weights: List[int], blur_kernel_size: int):
    """ Full image processing using only Computer Vision methods. Return binary classification mask """
    
    b, g, r = cv2.split(image)

    r_pre, g_pre, b_pre = map(preprocess_channel, (r, g, b))
    r_ridge, g_ridge, b_ridge = map(ridge_detect_channel, (r_pre, g_pre, b_pre))

    normalize = lambda x: (x - x.min()) / (x.max() - x.min()) * 255
    r_norm, g_norm, b_norm = map(normalize, (r_ridge, g_ridge, b_ridge))
    r_post, g_post, b_post = map(postprocess_channel, (r_norm, g_norm, b_norm), [ blur_kernel_size ] * 3) 

    mask = combine_channels(r_post, g_post, b_post, threshold=threshold, weights=channel_weights)

    return mask

def combine_channels(r, g, b, threshold: int, weights: List[float]):
    # TODO: maybe change later with normalization?
    assert sum(weights) <= 1.0

    result = np.average([r, g, b], weights=weights, axis=0)
    _, mask = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)
    return mask
    
def ridge_detect_channel(channel: np.ndarray):
    return sato(channel, black_ridges=False)
    
def preprocess_channel(channel: np.ndarray):
    image = cv2.equalizeHist(channel)
    image = cv2.bitwise_not(image)
    return image
    
def postprocess_channel(channel: np.ndarray, blur_kernel_size = 19):
    # assume: uint8
    return cv2.GaussianBlur(channel, (blur_kernel_size, blur_kernel_size), 0)

def plt_show_channels(r, g, b):
    plt.subplot(131)
    plt.title("Red channel")
    plt.imshow(r, cmap='gray', vmin=0, vmax=255)

    plt.subplot(132)
    plt.title("Green channel")
    plt.imshow(g, cmap='gray', vmin=0, vmax=255)

    plt.subplot(133)
    plt.title("Blue channel")
    plt.imshow(b, cmap='gray', vmin=0, vmax=255)
    
# --- STATISTICS ---

from sklearn.metrics import classification_report, confusion_matrix

def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

def calculate_mask_statistics(gold_standard, mask):
    # Assume vmin = 0, vmax = 255, type = uint8t
    assert gold_standard.shape == mask.shape

    cm = confusion_matrix(gold_standard.flatten(), mask.flatten())
    tn, fp, fn, tp = cm.ravel()

    cm_accuracy = tp / (tp + fp)
    cm_specificity = tn / (tn + fp)
    cm_sensitivity = tp / (tp + fn)
    
    cm_mean = (cm_specificity + cm_sensitivity) / 2
    cm_geo = geo_mean_overflow([cm_specificity, cm_sensitivity]) 

    return (cm_accuracy, cm_specificity, cm_sensitivity, cm_mean, cm_geo)
    
def print_statistics(gold_standard, mask):
    acc, spec, sens, mean_spec_sens, geo_spec_sens = calculate_mask_statistics(gold_standard, mask)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Sensitivity: {spec * 100:.2f}%")
    print(f"Specificity: {sens * 100:.2f}%")
    print(f"Mean(spec, sens): {mean_spec_sens * 100:.2f}%")
    print(f"GeoMean(spec, sens): {geo_spec_sens * 100:.2f}%")