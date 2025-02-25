import os

# Set the base directory for data (pointing to the data/Chip20240118 folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'Chip20240118'))

# Change file names as needed; here using .mat for orientation map
ORIENTATION_MAP_FILE = os.path.join(DATA_DIR, r'run0\M28D20240118R0OrientationP2.mat')
OPSIN_TIF_FILE = os.path.join(DATA_DIR, 'SL1EX570DM600EM632L13OD0_binned.tif')
GCAMP_TIF_FILE = os.path.join(DATA_DIR, 'SL1EX480DM505EM520L100OD16_binned.tif')

# Input dimensions and parameters
INPUT_HEIGHT = 512
INPUT_WIDTH = 512
NUM_FRAMES = 24
NUM_CHANNELS = 4  # Channels: raw imaging, opsin, gcamp, orientation

# Convolutional layer parameters (for the encoder)
CONV1_FILTERS = 64
CONV1_KERNEL = (7, 7)
CONV1_STRIDES = (2, 2)

CONV2_FILTERS = 128
CONV2_KERNEL = (3, 3)
CONV2_STRIDES = (1, 1)

POOL_SIZE = (2, 2)
POOL_STRIDES = (2, 2)

# ConvLSTM parameters (for our PyTorch implementation, these are used in our custom ConvLSTMCell)
CONVLSTM1_FILTERS = 128  # reduced from 256 for memory efficiency
CONVLSTM1_KERNEL = (3, 3)

CONVLSTM2_FILTERS = 64   # reduced from 128
CONVLSTM2_KERNEL = (3, 3)

# Upsampling parameters (using ConvTranspose2d in PyTorch)
UP1_FILTERS = 128
UP1_KERNEL = (3, 3)
UP1_STRIDES = (2, 2)

UP2_FILTERS = 64
UP2_KERNEL = (3, 3)
UP2_STRIDES = (2, 2)

UP3_FILTERS = 1
UP3_KERNEL = (3, 3)
UP3_STRIDES = (2, 2)

# Training parameters
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-4
