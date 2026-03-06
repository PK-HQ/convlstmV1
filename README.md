# convlstmV1

A ConvLSTM-based encoder model of macaque V1 predicting population-level 
cortical activity from visual and optogenetic inputs (Seidemann Lab, UT Austin).

The training set is built from condition-averaged widefield images, where for 
each condition the blank (baseline) is subtracted from the optostim condition. 
Two datasets are processed:

- **run2 (visual baseline):** `TypeCond==3 - TypeCond==0`, labeled by `GaborOrt` and `StimCon`
- **run3 (columnar optostim):** same subtraction, additionally labeled by `BitmapOrt` (from `ProjImg`)

## Folder Structure
```
Project_Root/
├── data/
│   └── Chip20240118/
│       ├── run2/
│       │   ├── M28D20240118R2StabIntgS004E010.mat
│       │   └── M28D20240118R2TS.mat
│       └── run3/
│           ├── M28D20240118R3StabIntgS004E010.mat
│           └── M28D20240118R3TS.mat
│       ├── M28D20240118R0OrientationP2.tif
│       ├── SL1EX570DM600EM632L13OD0_binned.tif
│       └── SL1EX480DM505EM520L100OD16_binned.tif
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_encoder.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Setup
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

Loads condition-averaged optostim data, selects a max V-contrast + V-opto sample, 
runs a forward pass, and plots predicted V1 activity alongside orientation, opsin, 
and GCaMP maps.

*Status: research prototype — training integration in progress.*
