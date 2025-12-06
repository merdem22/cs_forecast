# -*- coding: utf-8 -*-
"""
preprocessing.py

Contains the low-level logic for processing all data for a single participant
from your .mat file dataset.

This is called by dataset_builder.py
"""

import os
import glob
import math
import scipy.io
import numpy as np
import mne
from typing import List, Tuple, Optional
from scipy import signal

def preprocess_eeg_segment(eeg_data: np.ndarray, fs: int) -> np.ndarray:
    """
    Applies the preprocessing steps from your paper to an EEG segment.
    - 3rd-order causal Butterworth bandpass filter (4-40 Hz)
    - Z-score normalization per channel (to match the repo's style)
    """
    # 1. Apply 3rd-order Butterworth band-pass filter (4-40 Hz), causal (forward-only)
    sos = signal.butter(3, [4, 40], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfilt(sos, eeg_data.astype(np.float64), axis=1)

    # 2. Apply Z-score normalization per-channel
    # (seg - mean) / std
    mean = np.mean(filtered, axis=1, keepdims=True)
    std = np.std(filtered, axis=1, keepdims=True) + 1e-8 # Add epsilon for stability
    normalized_eeg = (filtered - mean) / std

    return normalized_eeg

def _extract_windows(
    eeg: np.ndarray,
    window_samples: int,
    stride_samples: Optional[int],
    max_windows: Optional[int],
) -> List[np.ndarray]:
    """Slice the continuous EEG into multiple fixed-length windows."""
    stride = stride_samples if stride_samples is not None else window_samples
    total = eeg.shape[1]
    windows = []
    start = 0
    while start + window_samples <= total:
        windows.append(eeg[:, start:start + window_samples])
        start += stride
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def build_dataset_for_participant(
    participant_path: str,
    fs: int = 256,
    window_samples: int = 1024,
    stride_samples: Optional[int] = None,
    max_windows_per_trial: Optional[int] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes all sessions for a single participant.

    For each 8-second .mat trial file, it extracts the first 4 seconds
    of EEG data as the input (X) and the file's single binary label
    as the target (y).
    """
    all_eeg: List[np.ndarray] = []
    all_labels: List[int] = []
    all_session_ids: List[str] = []
    all_trial_ids: List[str] = []
    all_window_ids: List[int] = []
    
    # Iterate through the session subfolders (S1, S2, S3)
    for session_name in sorted(os.listdir(participant_path)):
        session_path = os.path.join(participant_path, session_name)
        if not os.path.isdir(session_path) or not session_name.startswith('S'):
            continue
            
        # Find all .mat files in the session folder
        mat_files = glob.glob(os.path.join(session_path, '*.mat'))
        if not mat_files:
            continue

        for mat_file_path in mat_files:
            try:
                mat = scipy.io.loadmat(mat_file_path)
                
                # 1. Extract the single binary label for the whole trial
                label = int(mat['label'][0][0])
                
                # 2. Extract the 14 EEG channels
                # (Data is [1:15] which is index 1 through 14)
                eeg_data = mat['EEGData'][1:15, :]
                
                # 3. Apply filtering and normalization
                processed_eeg = preprocess_eeg_segment(eeg_data, fs=fs)
                
                trial_name = os.path.splitext(os.path.basename(mat_file_path))[0]

                windows = _extract_windows(
                    processed_eeg,
                    window_samples=window_samples,
                    stride_samples=stride_samples,
                    max_windows=max_windows_per_trial,
                )

                for w_idx, window in enumerate(windows):
                    if window.shape == (14, window_samples):
                        all_eeg.append(window)
                        all_labels.append(label)
                        all_session_ids.append(session_name)
                        all_trial_ids.append(trial_name)
                        all_window_ids.append(w_idx)
                
            except Exception as e:
                print(f"[preprocessing] Skipping file {mat_file_path}: {e}")
                continue

    if not all_eeg:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Stack all windows and labels into NumPy arrays
    eeg_windows = np.stack(all_eeg, axis=0)  # (N, C, T)
    labels = np.array(all_labels, dtype=np.int32)
    session_ids = np.array(all_session_ids, dtype=object)
    trial_ids = np.array(all_trial_ids, dtype=object)
    window_ids = np.array(all_window_ids, dtype=np.int32)

    return eeg_windows, labels, session_ids, trial_ids, window_ids
