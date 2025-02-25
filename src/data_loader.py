import numpy as np
import cv2
from config import ORIENTATION_MAP_FILE, OPSIN_TIF_FILE, GCAMP_TIF_FILE, INPUT_HEIGHT, INPUT_WIDTH
from scipy.io import loadmat


def rescale_array(arr, new_min, new_max):
    old_min = np.min(arr)
    old_max = np.max(arr)
    if old_max - old_min == 0:
        return np.full_like(arr, new_min)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def load_orientation_map():
    data = loadmat(ORIENTATION_MAP_FILE)
    if "RespCondPCA" not in data:
        raise ValueError("The .mat file must contain the 'RespCondPCA' variable.")
    # Load RespCondPCA, which is expected to be 512x512x12
    resp_pca = data["RespCondPCA"].astype(float)
    if resp_pca.ndim != 3 or resp_pca.shape[2] < 7:
        raise ValueError("RespCondPCA must have at least 7 slices (512x512x12 expected).")

    # In MATLAB, indices are 1-based: slice 1 is for 0-tuned and slice 7 for 90-tuned.
    # In Python, convert these to 0-based indices: 0 and 6.
    ort0 = rescale_array(resp_pca[:, :, 0], 0, 1)
    ort90 = rescale_array(resp_pca[:, :, 6], 0, 1)

    # Compute the difference: 90-tuned minus 0-tuned, and rescale to [-1, 1].
    diff_map = ort90 - ort0
    ort90_0 = rescale_array(diff_map, -1, 1)

    # Expand dims to ensure shape (512, 512, 1)
    if ort90_0.ndim == 2:
        ort90_0 = np.expand_dims(ort90_0, axis=-1)
    return ort90_0

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
