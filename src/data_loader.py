import numpy as np
import cv2
from config import ORIENTATION_MAP_FILE, OPSIN_TIF_FILE, GCAMP_TIF_FILE, INPUT_HEIGHT, INPUT_WIDTH
from scipy.io import loadmat

def load_orientation_map():
    data = loadmat(ORIENTATION_MAP_FILE)
    if "MapOrt" not in data:
        raise ValueError("The .mat file must contain the 'MapOrt' variable.")
    # Orientation map shape should be (512, 512) or (512,512,1)
    orientation_map = data["MapOrt"].astype(float)
    if orientation_map.ndim == 2:
        orientation_map = np.expand_dims(orientation_map, axis=-1)
    return orientation_map

def load_tif_map(filepath):
    # Read image in grayscale mode (shape: H x W)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {filepath}")
    # Convert to float32 and normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    return img

def load_all_data():
    orientation_map = load_orientation_map()  # from MAT file
    # Load opsin and GCaMP maps from their TIF files, now scaled to [0,1]
    opsin_map = load_tif_map(OPSIN_TIF_FILE)
    gcamp_map = load_tif_map(GCAMP_TIF_FILE)
    return orientation_map, opsin_map, gcamp_map

if __name__ == "__main__":
    orientation_map, opsin_map, gcamp_map = load_all_data()
    print("Orientation map shape:", orientation_map.shape)
    print("Opsin map shape:", opsin_map.shape)
    print("GCaMP map shape:", gcamp_map.shape)
