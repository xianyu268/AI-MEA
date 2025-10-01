import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d


def remove_nan(signal):
    return signal[~np.isnan(signal)]


def butter_filter(signal, cutoff, fs, order=4, filter_type='low', pad_length=100):
    padded_signal = apply_padding(signal, pad_length)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_padded_signal = filtfilt(b, a, padded_signal)
    return filtered_padded_signal[pad_length:-pad_length]


def plot_signal_comparison(original, filtered, resampled, title, output_folder, col_name):
    plt.figure(figsize=(10, 5))
    plt.plot(original, label="Original Signal", linestyle='--')
    plt.plot(filtered, label="Filtered Signal")
    plt.plot(resampled, label="Resampled Signal", linestyle=':')
    plt.title(title)
    plt.xlabel("Sample Points")
    plt.ylabel("Amplitude")
    plt.legend()
    # plt.show()

    # 保存图片到指定文件夹
    output_path = os.path.join(output_folder, f"{title}_{col_name}.png")
    plt.savefig(output_path)
    plt.close()


def savgolay_filter(signal, window_length, polyorder, pad_length=200):
    padded_signal = apply_padding(signal, pad_length)
    filtered_padded_signal = savgol_filter(padded_signal, window_length, polyorder)
    return filtered_padded_signal[pad_length:-pad_length]


def apply_padding(signal, pad_length, mode='reflect'):
    return np.pad(signal, (pad_length, pad_length), mode=mode)


def smooth_transition_area(signal, filtered_signal, peak_index, peak_window_length, transition_length=100):
    """
    对滤波信号的峰值前后部分进行平滑，以自然连接峰值段。
    """
    start_index = max(0, peak_index - peak_window_length)
    end_index = min(len(signal), peak_index + peak_window_length)
    smoothed_signal = np.copy(filtered_signal)

    transition_start = max(0, start_index - transition_length)
    transition_end = min(len(signal), end_index + transition_length)

    # 使用S形加权函数
    transition_weights_before = 1 / (1 + np.exp(-np.linspace(1, -1, start_index - transition_start) * 10))
    transition_weights_after = 1 / (1 + np.exp(-np.linspace(-1, 1, transition_end - end_index) * 10))

    for i, idx in enumerate(range(transition_start, start_index)):
        smoothed_signal[idx] = (
            transition_weights_before[i] * filtered_signal[idx] + (1 - transition_weights_before[i]) * signal[idx]
        )

    for i, idx in enumerate(range(end_index, transition_end)):
        smoothed_signal[idx] = (
            transition_weights_after[i] * filtered_signal[idx] + (1 - transition_weights_after[i]) * signal[idx - 1]
        )

    smoothed_signal[start_index:end_index] = signal[start_index:end_index]

    return smoothed_signal


def filter_with_peak_replacement(signal, filtered_signal, peak_ratio=0.05, transition_length=3):
    peak_index = np.argmax(signal)
    peak_window_length = int(len(signal) * peak_ratio)
    smoothed_filtered_signal = smooth_transition_area(signal, filtered_signal, peak_index,
                                                      peak_window_length, transition_length)
    return smoothed_filtered_signal


def extract_peak_window(signal, signal_type, target_length=10000, pre_peak=2500, post_peak=7500):
    """
    从信号中提取以绝对值最大点为中心的固定窗口，并填充至目标长度 target_length。

    :param signal: 输入信号
    :param signal_type: 信号类型 ("extracellular" 或 "intracellular")
    :param target_length: 最终目标长度（默认为 10000）
    :param pre_peak: 峰值前的采样点数
    :param post_peak: 峰值后的采样点数
    :return: 固定长度的信号窗口
    """
    # 找到最大值和最小值及其索引
    max_index = np.argmax(signal)
    min_index = np.argmin(signal)
    max_value = signal[max_index]
    min_value = signal[min_index]

    # 选择峰值位置
    if signal_type == 'extracellular':
        # 比较绝对值，选取绝对值大的为主峰
        main_peak = max_index if abs(max_value) >= abs(min_value) else min_index
    else:
        main_peak = max_index

    # 计算截取范围
    start = max(main_peak - pre_peak, 0)
    end = min(main_peak + post_peak, len(signal))
    windowed_signal = signal[start:end]

    # 填充不足部分到 target_length
    padding_before = max(0, pre_peak - main_peak)
    padding_after = target_length - len(windowed_signal) - padding_before
    windowed_signal = np.pad(windowed_signal, (padding_before, padding_after), mode='edge')

    # 确保截取的信号长度为 target_length
    return windowed_signal[:target_length]


def process_all_excels_in_folder(folder_path, csv_folder_path, output_folder, output_figure_folder,
                                 fs=1000, cutoff=10, filter_type='low', sg_window=11, sg_poly=2,
                                 filter_method='butter', pad_length=100, transition_length=50, visualize=False):
    intra_signals_list = []
    extra_signals_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            base_name = os.path.splitext(file_name)[0]
            intra_csv_path = os.path.join(csv_folder_path, f'{base_name}_intra_signals.csv')
            extra_csv_path = os.path.join(csv_folder_path, f'{base_name}_extra_signals.csv')

            if os.path.exists(intra_csv_path) and os.path.exists(extra_csv_path):
                intra_df = pd.read_csv(intra_csv_path)
                extra_df = pd.read_csv(extra_csv_path)

                for col in intra_df.columns:
                    if col in extra_df.columns:
                        intra_signal = remove_nan(intra_df[col].values)
                        extra_signal = remove_nan(extra_df[col].values)

                        if len(intra_signal) == 0 or len(extra_signal) == 0:
                            print(f"Skipping empty signal pair in column {col}")
                            continue

                        extra_filtered = extra_signal

                        # 滤波
                        if filter_method == 'butter':
                            intra_filtered = butter_filter(intra_signal, cutoff, fs, filter_type=filter_type,
                                                           pad_length=pad_length)
                        elif filter_method == 'savgol':
                            intra_filtered = savgolay_filter(intra_signal, sg_window, sg_poly, pad_length=pad_length)
                        elif filter_method == 'None':
                            intra_filtered = intra_signal
                        else:
                            raise ValueError('Please select filter method in butter, savgol, or None')

                        # 应用峰值平滑逻辑
                        intra_smoothed = filter_with_peak_replacement(intra_signal, intra_filtered,
                                                                      peak_ratio=0.03, transition_length=transition_length)

                        # 峰值窗口提取
                        intra_windowed = extract_peak_window(intra_smoothed, signal_type='intracellular')
                        extra_windowed = extract_peak_window(extra_signal, signal_type='extracellular')

                        if visualize:
                            plot_signal_comparison(
                                intra_signal, intra_filtered, intra_windowed,
                                f"Intra Signal Filtered and Windowed - {col}", output_figure_folder, col
                            )
                            # plot_signal_comparison(
                            #     extra_signal, extra_filtered, extra_windowed,
                            #     f"Extra Signal Filtered and Windowed - {col}", output_figure_folder, col
                            # )

                        intra_signals_list.append(intra_windowed)
                        extra_signals_list.append(extra_windowed)

    if intra_signals_list and extra_signals_list:
        y_data = np.array(intra_signals_list).T
        x_data = np.array(extra_signals_list).T
        x_train, x_test, y_train, y_test = train_test_split(x_data.T, y_data.T, test_size=0.2, random_state=60)
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

        train_output_path = os.path.join(output_folder, 'train_data_enhanced_CH31.npy')
        test_output_path = os.path.join(output_folder, 'test_data_R5V7.npy')
        np.save(train_output_path, [y_data, x_data])
        # np.save(test_output_path, [y_test, x_test])

        print(f"处理完成，数据已保存为 {train_output_path} 和 {test_output_path}")
    else:
        print("No consistent signal pairs found.")


# 设置文件路径
folder_path = r'./enhanced_signals'
csv_folder_path = './enhanced_signals/csv_file'
output_folder = './npy_file'
os.makedirs(output_folder, exist_ok=True)
output_figure_folder = './plot_enhanced'
os.makedirs(output_figure_folder, exist_ok=True)

process_all_excels_in_folder(
    folder_path, csv_folder_path, output_folder, output_figure_folder, fs=20000, cutoff=100, filter_type='low',
    sg_window=600, sg_poly=3, filter_method='savgol', pad_length=1000, transition_length=30, visualize=True
)
