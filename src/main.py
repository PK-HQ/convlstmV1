# src/main.py
"""
Main script for training the V1 encoder model using condition-averaged optostim data,
implemented in PyTorch.
For each optostim sample (from run3), we compute the difference:
    DataCond(TS.Header.Conditions.TypeCond==3) - DataCond(TS.Header.Conditions.TypeCond==0)
and (if necessary) average over the 11 frames to obtain a ground truth image of shape (512,512,1).
Each sample is paired with an input tensor built from static maps:
    - Dummy raw imaging (zeros)
    - Opsin map (flat dummy map)
    - GCaMP map (flat dummy map)
    - Orientation map (loaded from MAT file)
The input tensor is of shape (num_samples, NUM_FRAMES, NUM_CHANNELS, 512, 512)
in PyTorch format (channels-first for spatial processing).
We then train the encoder to predict the ground truth condition-averaged image.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_encoder import V1Encoder
from data_loader import load_all_data
from data_preprocessing import load_opto_data
from config import NUM_FRAMES, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNELS, LEARNING_RATE, EPOCHS, BATCH_SIZE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def prepare_static_map(map_array):
    """
    Convert a 2D static map (H, W) or (H, W, 1) into shape (1, 1, 1, H, W),
    where the channel dimension is at index 2.
    """
    if map_array.ndim == 2:
        map_array = np.expand_dims(map_array, axis=-1)  # becomes (H, W, 1)
    map_prepared = map_array[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, 1)
    map_prepared = np.transpose(map_prepared, (0, 1, 4, 2, 3))  # (1, 1, 1, H, W)
    return map_prepared


def create_dummy_input(orientation_map, opsin_map, gcamp_map):
    """
    Create a dummy input tensor for the encoder.
    Combines dummy raw imaging (zeros) with static maps (opsin, gcamp, orientation),
    repeated across NUM_FRAMES.
    Returns a PyTorch tensor of shape (1, NUM_FRAMES, NUM_CHANNELS, 512, 512)
    with dtype torch.float32.
    """
    # Dummy raw imaging: shape (1, NUM_FRAMES, 1, 512, 512)
    dummy_raw = np.zeros((1, NUM_FRAMES, 1, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)

    # Prepare each static map to have shape (1, 1, 1, 512, 512)
    orientation_prepared = prepare_static_map(orientation_map)
    opsin_prepared = prepare_static_map(opsin_map)
    gcamp_prepared = prepare_static_map(gcamp_map)

    # Repeat each map along the time dimension: shape becomes (1, NUM_FRAMES, 1, 512, 512)
    orientation_input = np.repeat(orientation_prepared, NUM_FRAMES, axis=1)
    opsin_input = np.repeat(opsin_prepared, NUM_FRAMES, axis=1)
    gcamp_input = np.repeat(gcamp_prepared, NUM_FRAMES, axis=1)

    # Concatenate along the channel dimension (axis=2) to get shape (1, NUM_FRAMES, 4, 512, 512)
    combined_input = np.concatenate([dummy_raw, opsin_input, gcamp_input, orientation_input], axis=2)
    print("Combined input shape:", combined_input.shape)
    return torch.tensor(combined_input, dtype=torch.float32)


def build_training_dataset(samples, orientation_map, opsin_map, gcamp_map):
    """
    Build training arrays from a list of optostim samples.
    For each sample, if the sample image has shape (512,512,11), average over axis 2;
    if it's 2D (512,512), use it directly.
    Then expand dims to (512,512,1) and transpose to PyTorch format (channels-first).
    Returns:
        X: tensor of shape (num_samples, NUM_FRAMES, NUM_CHANNELS, 512, 512)
        Y: tensor of shape (num_samples, 1, 512, 512)
    """
    num_samples = len(samples)
    dummy_input = create_dummy_input(orientation_map, opsin_map, gcamp_map)  # (1, NUM_FRAMES, 4, 512, 512)
    dummy_input = np.squeeze(dummy_input, axis=0)  # now shape: (NUM_FRAMES, 4, 512, 512)
    X = np.repeat(dummy_input[np.newaxis, ...], num_samples, axis=0)  # (num_samples, NUM_FRAMES, 4, 512, 512)

    Y_list = []
    for sample in samples:
        if sample['image'].ndim == 3:
            gt_image = np.mean(sample['image'], axis=2)  # (512,512)
        elif sample['image'].ndim == 2:
            gt_image = sample['image']  # (512,512)
        else:
            raise ValueError("Unexpected image dimensions for sample['image']")
        gt_image = np.expand_dims(gt_image, axis=-1)  # (512,512,1)
        gt_image = np.transpose(gt_image, (2, 0, 1))  # (1, 512, 512)
        Y_list.append(gt_image)
    Y = np.array(Y_list)  # (num_samples, 1, 512, 512)
    print("Training data shapes: X =", X.shape, ", Y =", Y.shape)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


# -------------------- Bandpass Filter Functions -------------------- #

def func_wo_n_fermi(parm, SF2D):
    """
    Compute the Fermi function on the spatial frequency grid.
    parm is a list/array: [scale, cutoff, offset, delta].
    Returns a filter with the same shape as SF2D.
    """
    scale, cutoff, offset, delta = parm
    # Fermi function: 1 / (1 + exp((SF2D - cutoff)/delta))
    return 1.0 / (1.0 + np.exp((SF2D - cutoff) / delta))


def filter_fermi_3d(image, LowCutOff, HighCutOff, SizePxl):
    """
    Apply a 3D bandpass Fermi filter to a 2D (or 3D stack) image.
    image: input image (H x W) or (H x W x NumSlices)
    LowCutOff, HighCutOff: cutoff frequencies (cycles per mm)
    SizePxl: size of one pixel in mm (imgMM / imgPxl)
    Returns the filtered image with the same shape as input.
    """
    # If image is 2D, reshape to (H, W, 1)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    H, W, NumSlices = image.shape

    # Fermi filter parameters
    ParmFermiLowPass = [1, HighCutOff, 0, HighCutOff * 0.05]
    ParmFermiHighPass = [1, LowCutOff, 0, LowCutOff * 0.05]

    # Generate spatial frequency grid
    SFX = ((np.arange(W) - np.floor(W / 2) - 1) / (W - 1)) / SizePxl
    SFY = ((np.arange(H) - np.floor(H / 2) - 1) / (H - 1)) / SizePxl
    SFXX, SFYY = np.meshgrid(SFX, SFY)
    SF2D = np.abs(SFXX + 1j * SFYY)

    # Create Fermi filters
    if np.isinf(HighCutOff):
        FiltFermiLowPass = np.zeros((H, W))
    else:
        FiltFermiLowPass = func_wo_n_fermi(ParmFermiLowPass, SF2D)

    if LowCutOff == 0:
        FiltFermiHighPass = np.ones((H, W))
    else:
        FiltFermiHighPass = func_wo_n_fermi(ParmFermiHighPass, SF2D)

    FiltFermi = FiltFermiHighPass - FiltFermiLowPass

    # Expand filter to 3D
    FiltFermi3D = np.repeat(FiltFermi[:, :, np.newaxis], NumSlices, axis=2)

    # Apply FFT to each slice (apply along axes 0 and 1)
    F = np.fft.fft2(image, axes=(0, 1))
    F = np.fft.fftshift(F, axes=(0, 1))

    # Multiply by filter
    F_filtered = F * FiltFermi3D

    # Inverse FFT to get filtered image
    F_filtered = np.fft.ifftshift(F_filtered, axes=(0, 1))
    B = np.fft.ifft2(F_filtered, axes=(0, 1))
    # If image was originally real, return real part
    if np.isrealobj(image):
        B = np.real(B)
    return B


# -------------------- End of Bandpass Filter Functions -------------------- #

def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


def main():
    # Toggle to use flat maps for opsin and gcamp (True = flat, False = load from TIF)
    use_flat_maps = True

    # Amplification factor for orientation map
    amplification_factor = 20

    # Load static maps from TIF/MAT files
    orientation_map, opsin_map, gcamp_map = load_all_data()

    # Amplify the orientation map to increase its influence
    orientation_map = orientation_map * amplification_factor

    # Toggle opsin and gcamp maps if using flat maps
    if use_flat_maps:
        opsin_map = np.ones((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
        gcamp_map = np.ones((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)

    # Load optostim training samples from run3
    opto_samples = load_opto_data()
    print("Number of opto samples:", len(opto_samples))

    # Build training dataset (X, Y) from optostim conditions
    X_train, Y_train = build_training_dataset(opto_samples, orientation_map, opsin_map, gcamp_map)
    X_train, Y_train = X_train.to(device), Y_train.to(device)

    # Create DataLoader with batch_size 1 (to save memory)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build the encoder model and move it to device
    model = V1Encoder().to(device)
    print(model)

    # Define optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = torch.nn.MSELoss()

    # Training loop (MVP: 5 epochs)
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)  # Expected output shape: (B, 1, 512, 512)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    torch.cuda.empty_cache()

    # Evaluate using a DataLoader with batch size 1 to avoid OOM issues
    model.eval()
    eval_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    predicted_activities = []
    with torch.no_grad():
        for inputs, targets in eval_loader:
            outputs = model(inputs)  # Process one sample at a time
            predicted_activities.append(outputs)
            torch.cuda.empty_cache()

    # Prepare static maps for plotting
    opsin_plot = np.squeeze(opsin_map)
    gcamp_plot = np.squeeze(gcamp_map)
    orientation_plot = np.squeeze(orientation_map)
    mask0 = (orientation_plot >= 0-15) & (orientation_plot <= 0+15)
    mask90 = (orientation_plot >= 90-15) & (orientation_plot <= 90+15)
    orientation_plot_0 = np.where(mask0, orientation_plot, np.nan)
    orientation_plot_90 = np.where(mask90, orientation_plot, np.nan)

    # Plot the static maps
    fig, axs = plt.subplots(2, 4,figsize=(18, 12))

    im0 = axs[0, 0].imshow(opsin_plot, cmap='viridis')
    axs[0, 0].set_title("Opsin Map")
    axs[0, 0].axis("off")
    plt.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(gcamp_plot, cmap='viridis')
    axs[0, 1].set_title("GCaMP Map")
    axs[0, 1].axis("off")
    plt.colorbar(im1, ax=axs[0, 1])

    im2 = axs[0, 2].imshow(orientation_plot_0, cmap='hsv')
    axs[0, 2].set_title("Amplified Orientation Map")
    axs[0, 2].axis("off")
    plt.colorbar(im2, ax=axs[0, 2])

    im2 = axs[1, 2].imshow(orientation_plot_90, cmap='hsv')
    axs[1, 2].set_title("Amplified Orientation Map")
    axs[1, 2].axis("off")
    plt.colorbar(im2, ax=axs[1, 2])


    # Plot predicted activity along with ground truth

    # Get condition indices
    # Assume opto_samples is the list returned by load_opto_data()
    max_contrast = max(sample['StimCon'] for sample in opto_samples)

    baseline_indices = []
    opto_0_indices = []
    opto_90_indices = []

    for idx, sample in enumerate(opto_samples):
        # Check that the sample is at maximum contrast.
        if sample['StimCon'] == max_contrast:
            # If BitmapOrt is None, assume it's the baseline condition (no columnar optostim)
            if sample.get('BitmapOrt') is None:
                baseline_indices.append(idx)
            elif sample.get('BitmapOrt') == 0:
                opto_0_indices.append(idx)
            elif sample.get('BitmapOrt') == 90:
                opto_90_indices.append(idx)

    print("Opto 0 condition indices (max contrast):", max(opto_0_indices))
    print("Opto 90 condition indices (max contrast):", max(opto_90_indices))

    # Apply bandpass filter to predicted image for plotting
    bandpassSF = [0.8, 3] # Set bandpassSF = [LowCutOff, HighCutOff] in cycles per mm
    imgPxl = 512  # pixels
    imgMM = 8.22  # mm (image spans 8.22 mm)
    SizePxl = imgMM / imgPxl  # mm per pixel

    # For visualization, use the first sample's prediction
    predicted_image = predicted_activities[0].cpu().numpy().squeeze()  # (512,512)
    ground_truth = Y_train[0].cpu().numpy().squeeze()  # (512,512)
    predicted_image_filtered = filter_fermi_3d(predicted_image, bandpassSF[0], bandpassSF[1], SizePxl)
    ground_truth_filtered = filter_fermi_3d(ground_truth, bandpassSF[0], bandpassSF[1], SizePxl)

    # Normalize images for plotting if needed
    predicted_norm = normalize_image(predicted_image)
    ground_truth_norm = normalize_image(ground_truth)
    predicted_filtered_norm = normalize_image(predicted_image_filtered)
    ground_truth_filtered_norm = normalize_image(ground_truth_filtered)
    diff_norm = predicted_norm - ground_truth_norm
    diff_filtered_norm = predicted_filtered_norm - ground_truth_filtered_norm

    fig, axs = plt.subplots(2, 3,figsize=(18, 12))
    im3 = axs[0, 0].imshow(ground_truth_norm, cmap='gray')
    axs[0, 0].set_title("Filtered Actual V1 Activity")
    axs[0, 0].axis("off")
    plt.colorbar(im3, ax=axs[0, 0])

    im4 = axs[0, 1].imshow(predicted_norm, cmap='gray')
    axs[0, 1].set_title("Normalized Predicted V1 Activity")
    axs[0, 1].axis("off")
    plt.colorbar(im4, ax=axs[0, 1])

    im5 = axs[0, 2].imshow(diff_norm, cmap='seismic', vmin=-1, vmax=1)
    axs[0, 2].set_title("Difference (Predicted - Ground Truth)")
    axs[0, 2].axis("off")
    plt.colorbar(im5, ax=axs[0, 2])

    im3 = axs[1, 0].imshow(ground_truth_filtered_norm, cmap='gray')
    axs[1, 0].set_title("Filtered Actual V1 Activity")
    axs[1, 0].axis("off")
    plt.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(predicted_filtered_norm, cmap='gray')
    axs[1, 1].set_title("Normalized Predicted V1 Activity")
    axs[1, 1].axis("off")
    plt.colorbar(im4, ax=axs[1, 1])

    im5 = axs[1, 2].imshow(diff_filtered_norm, cmap='seismic', vmin=-1, vmax=1)
    axs[1, 2].set_title("Difference Filtered (Predicted - Ground Truth)")
    axs[1, 2].axis("off")
    plt.colorbar(im5, ax=axs[1, 2])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
