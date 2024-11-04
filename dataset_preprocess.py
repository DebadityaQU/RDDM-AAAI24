import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import glob
"""
处理单个CSV文件
"""
def process_single_file(file_path):

    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取所需的列并重命名
    extracted_df = pd.DataFrame({
        #'ACC_X': df['AccelX'],
        'GYR_X': df['GYR_X'],
        'PPG': df['PPG']
    })
    
    # 将提取的列添加到原始数据框的末尾
    result_df = pd.concat([df, extracted_df], axis=1)
    
    # 生成新的文件名
    file_name = os.path.basename(file_path)
    new_file_name = f"test_{file_name.split('_')[2].split('.')[0]}_Signals.csv"
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
    
    # 保存新的CSV文件
    result_df.to_csv(new_file_path, index=False)
    
    return new_file_path

"""
批量处理CSV文件,将matlab去噪后的csv文件转换为新的文件
"""
def batch_process_csv(base_path='E:\\PythonCode\\RDDM-main\\dataset\\IMU+PPG'):

    # 构建文件匹配模式
    input_pattern = os.path.join(base_path, 'test_128Hz_*.csv')
    
    # 获取匹配模式的所有文件
    input_files = glob.glob(input_pattern)
    
    # 按文件名排序，确保按序处理
    #input_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    
    processed_files = []
    for file in input_files:
        new_file = process_single_file(file)
        processed_files.append(new_file)
        print(f"处理完成: {file} -> {new_file}")
    
    print("所有文件处理完毕")
    return processed_files

'''
input: test_{file_id}_Signals.csv 去噪后的文件，包括Time_s, PPG, GYR_X三列
output: PPG与GYR的训练集与测试集的 .npy数组形式
'''
def preprocess_batch_csv_to_npy_with_normalization(data_dir, save_path, num_files=40, window_size=4, original_sampling_rate=1000, target_sampling_rate=128, resample='up'):
    all_ppg_segments = []
    all_ecg_segments = []
    all_gyr_segments = []
    
    # Determine the resampling factor
    resample_factor = original_sampling_rate / target_sampling_rate
    
    # Check if resampling is possible
    if resample not in ['up', 'down','no']:
        raise ValueError("resample should be either 'up' or 'down'")
    
    # Process each file
    for i in range(1, num_files + 1):
        # Format file ID with leading zeros (e.g., '01', '02', ..., '70')
        file_id = f"{i:02}"
        
        # UP-22 ~ UP-30   512Hz
        if i > 91:
            original_sampling_rate = 512
            resample_factor = original_sampling_rate / target_sampling_rate
        
        # Input file path
        file_path = data_dir + f"/test_{file_id}_Signals.csv"
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        time = data['Time_s']
        ppg = data['PPG']       # Using 'PPG' as ppg signal
        # acc = data['ACC_X']    # Using 'ACC_X' as acc signal (optional)
        gyr = data['LC_BCG2']     # Using 'GYR_X' as gyr signal
        ecg = data['ECG']
        if resample == 'down' and resample_factor > 1:
            # Downsample by averaging over blocks of size 'resample_factor'
            ppg_resampled = ppg[::int(resample_factor)]
            # acc_resampled = acc[::int(resample_factor)]
            gyr_resampled = gyr[::int(resample_factor)]
            ecg_resampled = ecg[::int(resample_factor)]
            new_time = time[::int(resample_factor)]
        
        elif resample == 'up' and resample_factor < 1:
            # Define the new time vector for upsampling
            new_time = np.arange(time.iloc[0], time.iloc[-1], 1/target_sampling_rate)
            
            # Interpolate the signals to the new time vector
            ppg_interp = interp1d(time, ppg, kind='linear', fill_value='extrapolate')
            # acc_interp = interp1d(time, acc, kind='linear', fill_value='extrapolate')
            gyr_interp = interp1d(time, gyr, kind='linear', fill_value='extrapolate')
            ecg_interp = interp1d(time, ecg, kind='linear', fill_value='extrapolate')
            ppg_resampled = ppg_interp(new_time)
            # acc_resampled = acc_interp(new_time)
            gyr_resampled = gyr_interp(new_time)
            ecg_resampled = ecg_interp(new_time)
        
        else:
            ppg_resampled = ppg
            # acc_resampled = acc[::int(resample_factor)]
            gyr_resampled = gyr
            ecg_resampled = ecg
            new_time = time          
        
        # Subject-specific z-score normalization for ppg and acc
        ppg_mean = np.mean(ppg_resampled)
        ppg_std = np.std(ppg_resampled)
        ecg_mean = np.mean(ecg_resampled)
        ecg_std = np.std(ecg_resampled)
        #acc_mean = np.mean(acc)
        #acc_std = np.std(acc)
        gyr_mean = np.mean(gyr_resampled)
        gyr_std = np.std(gyr_resampled)
        ppg_normalized = (ppg_resampled - ppg_mean) / ppg_std
        ecg_normalized = (ecg_resampled - ecg_mean) / ecg_std
        gyr_normalized = (gyr_resampled - gyr_mean) / gyr_std
        # Segment the signals into 4-second windows with 10% overlap
        window_samples = target_sampling_rate * window_size  # 512 samples per window
        overlap = int(window_samples * 0.1)  # 51 samples overlap
        step = window_samples - overlap  # 461 samples step size
        
        for start in range(0, len(ppg_normalized) - window_samples + 1, step):
            end = start + window_samples
            all_ppg_segments.append(ppg_normalized[start:end])
            all_gyr_segments.append(gyr_normalized[start:end])
            all_ecg_segments.append(ecg_normalized[start:end])
    # Convert lists to numpy arrays
    all_ppg_segments = np.array(all_ppg_segments)
    all_ecg_segments = np.array(all_ecg_segments)
    all_gyr_segments = np.array(all_gyr_segments)
    # Split into training (80%) and testing (20%)
    split_idx = int(len(all_ppg_segments) * 0.8)
    
    ppg_train, ppg_test = all_ppg_segments[:split_idx], all_ppg_segments[split_idx:]
    ecg_train, ecg_test = all_ecg_segments[:split_idx], all_ecg_segments[split_idx:]
    gyr_train, gyr_test = all_gyr_segments[:split_idx], all_gyr_segments[split_idx:]

    # np.savetxt(save_path+'/ppg_train.csv', ppg_train, delimiter=',')
    # np.savetxt(save_path+'/ppg_test.csv', ppg_test, delimiter=',')
    # np.savetxt(save_path+'/gyr_train.csv', gyr_train, delimiter=',')
    # np.savetxt(save_path+'/gyr_test.csv', gyr_test, delimiter=',')
    np.savetxt(save_path+'/ecg_train.csv', ecg_train, delimiter=',')
    np.savetxt(save_path+'/ecg_test.csv', ecg_test, delimiter=',')
    # Save to .npy files 
    # np.save(f"{save_path}/ppg_train_{window_size}sec.npy", ppg_train)
    # np.save(f"{save_path}/gyr_train_{window_size}sec.npy", gyr_train)
    # np.save(f"{save_path}/ppg_test_{window_size}sec.npy", ppg_test)
    # np.save(f"{save_path}/gyr_test_{window_size}sec.npy", gyr_test)
    np.save(f"{save_path}/ecg_test_{window_size}sec.npy", ecg_test)
    np.save(f"{save_path}/ecg_train_{window_size}sec.npy", ecg_train)

if __name__ == "__main__":
   # batch_process_csv()
    data_dir = 'dataset/IMU+PPG/BCG_dataset'
    save_path = 'dataset/IMU+PPG/BCG_dataset'
    preprocess_batch_csv_to_npy_with_normalization(data_dir, save_path,resample='down')