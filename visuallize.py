import matplotlib.pyplot as plt

'''
eval = True 表示评估模式
'''

import neurokit2 as nk
import numpy as np
import pandas as pd

'''
ecg的可视化
'''

def process_and_plot_ecg(signal, index,  output_dir="./figure", identifier=''):
        # Ensure signal is a 1D numpy array
        signal = np.array(signal).flatten()
        
        # Process signal data
        signal_signals, signal_info = nk.ecg_process(signal, sampling_rate=128)
        
        # Plot signal data
        nk.ecg_plot(signal_signals, signal_info)
        
        # Adjust figure size
        fig = plt.gcf()
        fig.set_size_inches(14, 10, forward=True)
        
        # Determine the filename based on the eval parameter
        filename = f"{output_dir}/{identifier}_ecg[{index}].png"
        
        # Save the figure
        fig.savefig(filename)
        
        # Close the figure to free up memory
        plt.close(fig)
        
        print(f"Successfully processed and saved plot: {filename}")

#ppg与imu的可视化，type表示要绘制的曲线类型
def process_and_plot(signal, ppg_roi, index, type='PPG', output_dir="./figure",identifier=''):
        # Ensure signal is a 1D numpy array
        signal = np.array(signal).flatten()
        
        # Check if ppg_roi is not None and has the same length as signal
        if ppg_roi is not None and len(ppg_roi) != len(signal):
            print(f"Warning: ppg_roi length ({len(ppg_roi)}) does not match signal length ({len(signal)})")
            ppg_roi = None  # Set to None to avoid issues in nk.ppg_plot
        
        # Process signal data
        signal_signals, signal_info = nk.ppg_process(signal, sampling_rate=128, static=True, denoise=False)
        
        # Plot signal data
        nk.ppg_plot(signal_signals, signal_info, roi_array=ppg_roi, title=type)
        
        # Adjust figure size
        fig = plt.gcf()
        fig.set_size_inches(14, 10, forward=True)
        
        # Determine the filename based on the eval parameter
        filename = f"{output_dir}/{identifier}_{type.lower().replace(' ', '_')}[{index}].png"
        
        # Save the figure
        fig.savefig(filename)
        
        # Close the figure to free up memory
        plt.close(fig)
        
        print(f"Successfully processed and saved plot: {filename}")
        

def visualize_diffusion(x, x_t, x_t_unmasked, pred_x_t,  _ts):
        """仅可视化前5个样本的加噪和去噪过程"""

        # 限制可视化样本数量为前5个
        num_samples = min(5, x.shape[0])

        plt.figure(figsize=(15, 12))
        for i in range(num_samples):
            # 原始信号
            plt.subplot(3, num_samples, i + 1)
            plt.plot(x[i, 0].cpu().numpy(), label=f'Original Sample {i+1}')
            plt.title(f'Original Sample {i+1}')

            # 加噪和去噪信号 (x_t 和 x_t_unmasked)
            plt.subplot(3, num_samples, num_samples + i + 1)
            plt.plot(x_t[i, 0].detach().cpu().numpy(), color='blue', label=f'Noisy (Masked) {i+1} (t={_ts[i]})', alpha=0.7,linewidth=2, linestyle='-')
            plt.plot(x_t[i, 0].detach().cpu().numpy(), color='orange', label=f'Noisy (Unmasked) {i+1} (t={_ts[i]})', alpha=0.7,linewidth=1, linestyle='--')
            plt.title(f'Noisy Samples {i+1} (t={_ts[i]})')
            plt.legend()  # 添加图例区分

            # 去噪后的信号
            plt.subplot(3, num_samples, 2 * num_samples + i + 1)
            plt.plot(pred_x_t[i, 0].detach().cpu().numpy(), label=f'Denoised Sample {i+1}')
            plt.title(f'Denoised Sample {i+1}')

        plt.tight_layout()
        plt.show()


