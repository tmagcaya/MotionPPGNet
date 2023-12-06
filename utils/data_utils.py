import glob
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import time
import pandas as pd
import math

from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter


def LoadTroikaDataset(data_dir):
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the
            reference data for data_fls[5], etc...
    """
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls

def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]

def bandpass_filter(signal, fs=125, order=4, window=[6/60, 1080/60]):
    """filter the signal between used to be 3rd order 40 and 240 BPM

    Args:
        signal: input signal
        fs: sampling freq

    Returns:
        new signal  withing the range of the desired frequencies
    """

    b, a = scipy.signal.butter(order, window, btype='bandpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def calculate_magnitude(x, y, z):
    """
    Calculate the magnitude for the signal

    Args:
        x: x-axis
        y: y-axis
        z: z-axis

    Returns:
        Signal magnitude value
    """
    return np.sqrt(x**2 + y**2 + z**2)