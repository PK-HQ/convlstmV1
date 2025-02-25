# src/data_preprocessing.py
"""
Data preprocessing module to load and process condition-averaged images for training.
It processes visual-only baseline and columnar optostim conditions.
"""

import numpy as np
import scipy.io
import os
from config import DATA_DIR


def load_mat_file(filepath, variable_name):
    data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    if variable_name not in data:
        raise ValueError(f"The .mat file must contain the '{variable_name}' variable.")
    return data[variable_name]


def load_TS(ts_filepath):
    TS = scipy.io.loadmat(ts_filepath, struct_as_record=False, squeeze_me=True)['TS']
    return TS


def load_baseline_data():
    """
    Load visual-only baseline condition data from run2.
    Compute: DataCond(TS.Header.Conditions.TypeCond==3) - DataCond(TS.Header.Conditions.TypeCond==0)
    and associate each difference image with GaborOrt and StimCon.
    Returns:
      List of dictionaries with keys: 'image', 'GaborOrt', 'StimCon'.
    """
    run2_dir = os.path.join(DATA_DIR, 'run2')
    data_filepath = os.path.join(run2_dir, 'M28D20240118R2StabIntgS004E010.mat')
    ts_filepath = os.path.join(run2_dir, 'M28D20240118R2TS.mat')

    DataCond = load_mat_file(data_filepath, 'DataCond')  # Expected shape: (512,512,N)
    TS = load_TS(ts_filepath)

    TypeCond = TS.Header.Conditions.TypeCond  # 1D array
    GaborOrt = TS.Header.Conditions.GaborOrt  # same length as TypeCond
    StimCon = TS.Header.Conditions.StimCon  # same length

    opt_idx = np.where(TypeCond == 3)[0]
    blank_idx = np.where(TypeCond == 0)[0]

    if len(blank_idx) == 0:
        raise ValueError("No blank condition found in baseline data.")
    blank_image = np.mean(DataCond[:, :, blank_idx], axis=2)

    baseline_samples = []
    for i in opt_idx:
        diff_image = DataCond[:, :, i] - blank_image
        sample = {
            'image': diff_image,  # (512,512)
            'GaborOrt': GaborOrt[i],
            'StimCon': StimCon[i]
        }
        baseline_samples.append(sample)
    return baseline_samples


def load_opto_data():
    """
    Load columnar optostim condition data from run3.
    Compute: DataCond(TS.Header.Conditions.TypeCond==3) - DataCond(TS.Header.Conditions.TypeCond==0)
    and associate each difference image with GaborOrt, StimCon, and BitmapOrt (derived from ProjImg).
    Returns:
      List of dictionaries with keys: 'image', 'GaborOrt', 'StimCon', 'BitmapOrt'.
    """
    run3_dir = os.path.join(DATA_DIR, 'run3')
    data_filepath = os.path.join(run3_dir, 'M28D20240118R3StabIntgS004E010.mat')
    ts_filepath = os.path.join(run3_dir, 'M28D20240118R3TS.mat')

    DataCond = load_mat_file(data_filepath, 'DataCond')
    TS = load_TS(ts_filepath)

    TypeCond = TS.Header.Conditions.TypeCond
    GaborOrt = TS.Header.Conditions.GaborOrt
    StimCon = TS.Header.Conditions.StimCon
    ProjImg = TS.Header.Conditions.ProjImg  # Assume this is a list/array of strings

    opt_idx = np.where(TypeCond == 3)[0]
    blank_idx = np.where(TypeCond == 0)[0]

    if len(blank_idx) == 0:
        raise ValueError("No blank condition found in opto data.")
    blank_image = np.mean(DataCond[:, :, blank_idx], axis=2)

    opto_samples = []
    for i in opt_idx:
        diff_image = DataCond[:, :, i] - blank_image
        proj_str = ProjImg[i]  # String
        if "O00000" in proj_str:
            BitmapOrt = 0
        elif "O09000" in proj_str:
            BitmapOrt = 90
        else:
            BitmapOrt = None
        sample = {
            'image': diff_image,
            'GaborOrt': GaborOrt[i],
            'StimCon': StimCon[i],
            'BitmapOrt': BitmapOrt
        }
        opto_samples.append(sample)

    for idx, sample in enumerate(opto_samples):
        print(f"Sample {idx} keys: {list(sample.keys())}")
        print(f"Sample {idx} contents: {sample}\n")

    return opto_samples


if __name__ == "__main__":
    baseline_samples = load_baseline_data()
    print("Number of baseline samples:", len(baseline_samples))

    opto_samples = load_opto_data()
    print(opto_samples)
    print("Number of opto samples:", len(opto_samples))
