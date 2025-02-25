# V1 Encoder Model Project

This project implements an encoder model for predicting V1 activity from visual and optogenetic inputs. The training set is built from condition-averaged images, where for each condition the blank (baseline) is subtracted from the optostim condition. Two separate datasets are processed:
- **Visual-only baseline condition (run2):** Uses `DataCond(TS.Header.Conditions.TypeCond==3) - DataCond(TS.Header.Conditions.TypeCond==0)` and labels each image with `GaborOrt` and `StimCon`.
- **Columnar optostim condition (run3):** Similarly, subtracts the blank condition and labels each image with `GaborOrt`, `StimCon`, and `BitmapOrt` (derived from `ProjImg`).

For a test run, the network simulates predicted activity for a condition (max V-contrast with V-opto stimulation) using static maps (orientation, opsin, GCaMP) and dummy raw data. The predicted output and the static maps are then plotted for visual inspection.

## Folder Structure
Project_Root/ ├── data/ │ └── Chip20240118/ │ ├── run2/ │ │ ├── M28D20240118R2StabIntgS004E010.mat │ │ └── M28D20240118R2TS.mat │ └── run3/ │ ├── M28D20240118R3StabIntgS004E010.mat │ └── M28D20240118R3TS.mat │ ├── M28D20240118R0OrientationP2.tif │ ├── SL1EX570DM600EM632L13OD0_binned.tif │ └── SL1EX480DM505EM520L100OD16_binned.tif ├── src/ │ ├── init.py │ ├── config.py │ ├── data_loader.py │ ├── data_preprocessing.py │ ├── model_encoder.py │ └── main.py ├── requirements.txt └── README.md

## Setup and Usage

1. Install dependencies:
    ```bash
    pip install python==3.11
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    pip install -r requirements.txt
    ```

2. Run the main script:
    ```bash
    python src/main.py
    ```

The script will load the condition-averaged optostim data, select a sample with maximum V-contrast and V-opto stimulation, run a forward pass through the untrained encoder model, and plot:
- Opsin map, GCaMP map, Orientation map, and Predicted V1 activity in a 2x2 subplot.
- The ground truth condition-averaged image for comparison.

Future work includes integrating the training routine with real imaging data.
