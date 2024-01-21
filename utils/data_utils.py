import glob
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import time
import pandas as pd
import math
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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

class TroikaDataset(Dataset):
    def __init__(self, data_files, ref_files, window_length, window_shift, fs):
        self.ppg_data = []
        self.acc_data = []
        self.labels = []

        for data_fl, ref_fl in zip(data_files, ref_files):
            ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)

            # Apply bandpass filter
            ppg = bandpass_filter(ppg, fs)
            accx = bandpass_filter(accx, fs)
            accy = bandpass_filter(accy, fs)
            accz = bandpass_filter(accz, fs)

            # Calculate accelerometer magnitude
            acc = calculate_magnitude(accx, accy, accz)

            # Normalize the signals
            ppg = (ppg - np.mean(ppg)) / np.std(ppg)
            acc = (acc - np.mean(acc)) / np.std(acc)

            # Load ground truth BPM
            ground_truth = scipy.io.loadmat(ref_fl)['BPM0'].reshape(-1)

            # Create windows for PPG and accelerometer data
            for i in range(0, len(ppg) - window_length + 1, window_shift):
                ppg_window = ppg[i:i + window_length]
                acc_window = acc[i:i + window_length]
                self.ppg_data.append(ppg_window)
                self.acc_data.append(acc_window)
                self.labels.append(ground_truth[i // window_shift])

        # Convert to numpy arrays and reshape for PyTorch
        self.ppg_data = np.array(self.ppg_data).astype('float32').reshape(-1, window_length, 1)
        self.acc_data = np.array(self.acc_data).astype('float32').reshape(-1, window_length, 1)
        self.labels = np.array(self.labels).astype('float32')

    def __len__(self):
        return len(self.ppg_data)

    def __getitem__(self, idx):
        ppg = self.ppg_data[idx]
        acc = self.acc_data[idx]
        label = self.labels[idx]

        # Permute the dimensions to match (batch, channels, sequence)
        ppg = torch.tensor(ppg).permute(1, 0)
        acc = torch.tensor(acc).permute(1, 0)
        label = torch.tensor(label).unsqueeze(0)

        return ppg, acc, label