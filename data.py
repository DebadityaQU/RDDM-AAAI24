import torch
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
import sklearn.preprocessing as skp
from torch.utils.data import Dataset, DataLoader

class IMUDataset(Dataset):
    def __init__(self, imu_data, ppg_data):
        self.imu_data = imu_data
        self.ppg_data = ppg_data

    def __getitem__(self, index):

        imu = self.imu_data[index]
        ppg = self.ppg_data[index]
        
        window_size = ppg.shape[-1]

        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        imu = nk.ppg_clean(imu.reshape(window_size), sampling_rate=128)
        _, info = nk.ppg_peaks(ppg.reshape(window_size), sampling_rate=128, method="elgendi")

        # Create a numpy array for ROI regions with the same shape as ECG
        ppg_roi_array = np.zeros_like(ppg.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["PPG_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ppg_roi_array[0, roi_start:roi_end] = 1

        return ppg.reshape(1, window_size).copy(), imu.reshape(1, window_size).copy(), ppg_roi_array.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecg_data)


class ECGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data

    def __getitem__(self, index):

        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        
        window_size = ecg.shape[-1]

        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        # Create a numpy array for ROI regions with the same shape as ECG
        ecg_roi_array = np.zeros_like(ecg.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["ECG_R_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ecg_roi_array[0, roi_start:roi_end] = 1

        return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecg_data)

def get_datasets(
    DATA_PATH = "./dataset/", 
    datasets=["bcg"],
    window_size=4,
    source = 'ppg',
    dest = 'ecg'
    ):

    dest_train_list = []
    source_train_list = []
    dest_test_list = []
    source_test_list = []
    
    for dataset in datasets:

        dest_train = np.load(DATA_PATH + dataset + f"/{dest}_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        source_train = np.load(DATA_PATH + dataset + f"/{source}_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        
        dest_test = np.load(DATA_PATH + dataset + f"/{dest}_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        source_test = np.load(DATA_PATH + dataset + f"/{source}_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)

        dest_train_list.append(dest_train)
        source_train_list.append(source_train)
        dest_test_list.append(dest_test)
        source_test_list.append(source_test)

    dest_train = np.nan_to_num(np.concatenate(dest_train_list).astype("float32"))
    source_train = np.nan_to_num(np.concatenate(source_train_list).astype("float32"))

    dest_test = np.nan_to_num(np.concatenate(dest_test_list).astype("float32"))
    source_test = np.nan_to_num(np.concatenate(source_test_list).astype("float32"))
    
    # Splitting test data into validation and test sets (50% each)
    num_test_samples = dest_test.shape[0]
    split_idx = num_test_samples // 2
    
    dest_val, dest_test = dest_test[:split_idx], dest_test[split_idx:]
    source_val, source_test = source_test[:split_idx], source_test[split_idx:]
    if dest == 'ecg':        
        dataset_train = ECGDataset(
            skp.minmax_scale(dest_train, (-1, 1), axis=1),
            skp.minmax_scale(source_train, (-1, 1), axis=1)
        )
        
        dataset_val = ECGDataset(
        skp.minmax_scale(dest_val, (-1, 1), axis=1),
        skp.minmax_scale(source_val, (-1, 1), axis=1)
        )
        
        dataset_test = ECGDataset(
            skp.minmax_scale(dest_test, (-1, 1), axis=1),
            skp.minmax_scale(source_test, (-1, 1), axis=1)
        )
    else:
        dataset_train = IMUDataset(
            skp.minmax_scale(dest_train, (-1, 1), axis=1),
            skp.minmax_scale(source_train, (-1, 1), axis=1)
        )
        dataset_val = IMUDataset(
        skp.minmax_scale(dest_val, (-1, 1), axis=1),
        skp.minmax_scale(source_val, (-1, 1), axis=1)
        )
        dataset_test = IMUDataset(
            skp.minmax_scale(dest_test, (-1, 1), axis=1),
            skp.minmax_scale(source_test, (-1, 1), axis=1)
        )
    return dataset_train, dataset_val, dataset_test