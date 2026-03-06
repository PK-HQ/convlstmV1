# convlstmV1

A ConvLSTM-based encoder model of macaque V1 that predicts 
population-level cortical activity from visual and optogenetic inputs. 
Built for closed-loop visual prosthetics research (Seidemann Lab, UT Austin).

## Overview

The model takes as input:
- **Visual stimuli** (Gabor orientation, contrast)
- **Optogenetic stimulation patterns** (columnar bitmap, opsin expression map)
- **Static cortical maps** (orientation map, GCaMP expression map)

And predicts widefield V1 population activity, enabling simulation of 
stimulation-evoked responses prior to live experiments.

## Folder Structure

Project_Root/
├── data/
│   └── Chip20240118/
│       ├── run2/                        # Visual-only baseline sessions
│       │   ├── M28D20240118R2StabIntgS004E010.mat
│       │   └── M28D20240118R2TS.mat
│       └── run3/                        # Columnar optostim sessions
│           ├── M28D20240118R3StabIntgS004E010.mat
│           └── M28D20240118R3TS.mat
│       ├── M28D20240118R0OrientationP2.tif   # Orientation map
│       ├── SL1EX570DM600EM632L13OD0_binned.tif   # Opsin map
│       └── SL1EX480DM505EM520L100OD16_binned.tif  # GCaMP map
├── src/
│   ├── __init__.py
│   ├── config.py                        # Paths, hyperparameters
│   ├── data_loader.py                   # Loads .mat sessions + static maps
│   ├── data_preprocessing.py            # Condition averaging, blank subtraction
│   ├── model_encoder.py                 # ConvLSTM encoder architecture
│   └── main.py                          # Entry point
├── requirements.txt
└── README.md

## Setup
```bash
pip install -r requirements.txt
```

> Requires Python 3.11, PyTorch (CUDA 12.1). See requirements.txt for full list.

## Usage
```bash
python src/main.py
```

Runs a forward pass on a sample condition (max V-contrast + V-opto), 
plots predicted V1 activity alongside orientation, opsin, and GCaMP maps.

## Data Format

Training data is built from condition-averaged widefield images:
- **run2** (visual baseline): `TypeCond==3` minus `TypeCond==0`, 
  labeled by `GaborOrt` and `StimCon`
- **run3** (columnar optostim): same subtraction, additionally 
  labeled by `BitmapOrt` derived from `ProjImg`

## Status

Research prototype — training integration with full imaging dataset in progress.
