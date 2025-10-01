import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# 自动创建路径并保存CSV文件和图像
def create_output_directory_and_files(output_file, plot_path):
    output_dir = os.path.dirname(output_file)
    plot_dir = os.path.dirname(plot_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    return output_file, plot_path


# 匹配胞内和胞外信号，并保存图像
def plot_intra_and_extra(intra_signals, extra_signals, channel_name, save_dir, excel_prefix):
    if intra_signals is None or extra_signals is None:
        return

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(extra_signals, label=f'Extracellular {channel_name}')
    plt.title(f'Extracellular: {channel_name}')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(intra_signals, label=f'Intracellular {channel_name}')
    plt.title(f'Intracellular: {channel_name}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()

    mixed_plot_dir = os.path.join(save_dir, f'{excel_prefix}_mixed_plots')
    if not os.path.exists(mixed_plot_dir):
        os.makedirs(mixed_plot_dir)

    plot_file = os.path.join(mixed_plot_dir, f"{excel_prefix}_{channel_name}_intra_extra.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved matched plot for {channel_name} at {plot_file}")


def normalize_data(signal_data):
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    if max_val == min_val:
        return signal_data
    return (signal_data - min_val) / (max_val - min_val)


def calculate_signal_std(signal_data):
    normalized_data = normalize_data(signal_data)
    return np.std(normalized_data)


def mean_filter(dat, parameter):
    n = parameter
    template = np.ones(n) / n
    datf = np.full((n - 1) // 2, dat[0])
    datb = np.full((n - 1) // 2, dat[-1])
    datLong = np.concatenate([datf, dat, datb])
    datLong = np.convolve(datLong, template, mode='same')
    datRes = datLong[n - 1:n - 1 + len(dat)]

    return datRes


def find_pacing_points_and_apd50(segmented_signals, fs, ap_up_wind_len=0.25):
    pacing_point = []
    apd50_value = []
    extra_apd_value = []
    repolarization_time = []
    left_points = []
    right_points = []
    mean_window = None
    std_window = None

    for signal in segmented_signals:
        signal_np = signal.to_numpy() if isinstance(signal, pd.Series) else signal
        print(signal_np.shape)

        filtered_signal = mean_filter(signal_np, 5)
        filtered_signal = mean_filter(filtered_signal, 11)
        filtered_signal = mean_filter(filtered_signal, 15)

        max_index = np.argmax(filtered_signal)
        max_value = filtered_signal[max_index]

        window_length = int(ap_up_wind_len * len(filtered_signal))
        start_idx = max(0, max_index - window_length)

        candidate_start = None
        baseline_value = np.mean(filtered_signal[:500])
        baseline = np.ones(20000) * baseline_value

        for idx in range(start_idx, max_index):
            process_point = filtered_signal[idx]
            window_data = np.append(filtered_signal[0:idx], baseline)

            mean_window = np.mean(window_data)
            std_window = np.std(window_data)

            if std_window < 0.5:
                ap_up_detect = 35
                ap_up_determine = 30
            elif 0.5 < std_window < 1:
                ap_up_detect = 30
                ap_up_determine = 12
            elif 1 < std_window < 2:
                ap_up_detect = 20
                ap_up_determine = 11
            elif 1 < std_window < 3 and mean_window < 0:
                ap_up_detect = 30
                ap_up_determine = 20
            elif mean_window < std_window < 2 * mean_window and std_window > 10:
                ap_up_detect = 3
                ap_up_determine = 2
            elif std_window > 2 * mean_window and std_window > 10:
                ap_up_detect = 3
                ap_up_determine = 1
            else:
                ap_up_detect = 10
                ap_up_determine = 4

            if process_point > mean_window + ap_up_detect * std_window:
                candidate_start = idx
                break

        start_point = None
        if candidate_start is not None:
            for idx in range(candidate_start, 0, -1):
                if filtered_signal[idx] < mean_window + ap_up_determine * std_window:
                    start_point = idx
                    break

        if start_point is None:
            if start_idx < max_index:
                sub_signal = filtered_signal[start_idx:max_index]
                if len(sub_signal) > 0:
                    start_point = np.argmin(sub_signal) + start_idx
                else:
                    print("Filtered signal in range is empty, skipping this segment.")
                    continue
            else:
                print("Start index is greater than or equal to max index, skipping this segment.")
                continue

        if start_point is not None:
            print('起搏点的值为：', filtered_signal[start_point])
            half_max_value = (max_value + signal_np[start_point]) / 2
            left_idx = np.where(signal_np[:max_index] <= half_max_value)[0]
            right_idx = np.where(signal_np[max_index:] <= half_max_value)[0]

            if len(left_idx) > 0 and len(right_idx) > 0:
                apd50 = (right_idx[0] + max_index) - left_idx[-1]
                pacing_point.append(start_point)
                apd50_value.append(apd50 / fs)

                extra_apd_start = signal_np[start_point]
                extra_apd_idx = np.where(signal_np[max_index:] == extra_apd_start)[0]
                if len(extra_apd_idx) > 0:
                    extra_apd = extra_apd_idx[0] + max_index - start_point
                    extra_apd_value.append(extra_apd / fs)
                else:
                    extra_apd_value.append(np.nan)

                min_index_after_max = np.argmin(filtered_signal[max_index:]) + max_index

                if min_index_after_max > max_index:
                    repolarization_duration = (min_index_after_max - max_index) / fs
                    repolarization_time.append(repolarization_duration)

                left_points.append(left_idx[-1])
                right_points.append(right_idx[0] + max_index)

    return pacing_point, apd50_value, extra_apd_value, repolarization_time, left_points, right_points


def plot_signal_with_segments(signal_series, peaks, avg_interval, pre_peak_ratio, post_peak_ratio, channel_name,
                              signal_type, plot_dir, excel_prefix):
    """
    绘制分段信号并保存。
    Args:
        signal_series (pd.Series): 原始信号序列。
        peaks (list): 检测到的峰值索引。
        avg_interval (float): 峰值间平均间隔。
        pre_peak_ratio (float): 峰值前的分段比例。
        post_peak_ratio (float): 峰值后的分段比例。
        channel_name (str): 信号通道名称。
        signal_type (str): 信号类型（胞内/胞外）。
        plot_dir (str): 保存图片的根目录。
        excel_prefix (str): 数据文件的前缀名。
    """
    segmented_plot_dir = os.path.join(plot_dir, f'{excel_prefix}_segmentation_plots')
    if not os.path.exists(segmented_plot_dir):
        os.makedirs(segmented_plot_dir)
        print(f"Created directory for segmented plots: {segmented_plot_dir}")

    plt.figure(figsize=(12, 6))
    plt.plot(signal_series, label='Original Signal', color='lightgray')

    segmented_signals = []
    all_pacing_points = []
    all_apd50_values = []
    all_left_points = []
    all_right_points = []

    for i, peak in enumerate(peaks):
        start_idx = int(max(0, peak - pre_peak_ratio * avg_interval))
        end_idx = int(min(len(signal_series) - 1, peak + post_peak_ratio * avg_interval))

        segmented_signal = signal_series[start_idx:end_idx]
        segmented_signals.append(segmented_signal)

        plt.plot(range(start_idx, end_idx), segmented_signal, label=f'Segment {i + 1} at {start_idx}-{end_idx}')

        if signal_type == 'intracellular':
            pacing_points, apd50_values, apd_values, repolarization_time, left_points, right_points = find_pacing_points_and_apd50(
                [segmented_signal], fs=20000)

            all_pacing_points.extend(pacing_points)
            all_apd50_values.extend(apd50_values)
            all_left_points.extend(left_points)
            all_right_points.extend(right_points)

    plt.title(f"{signal_type.capitalize()} Signal Segmentation: {channel_name}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend().set_visible(False)

    plot_path = os.path.join(segmented_plot_dir, f'{excel_prefix}_{channel_name}_{signal_type}_segmented.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved segmentation plot for {channel_name} ({signal_type}) at {plot_path}")

    return segmented_signals, all_pacing_points, all_apd50_values, all_left_points, all_right_points


def extract_signals(df, signal_type, signal_dict, excel_prefix, save_dir, default_min_peak_distance=4000):
    channel_start_col = 1
    min_peak_distances = {}

    print(f"Extracting {signal_type} signals from DataFrame with shape {df.shape}")


    for col in range(channel_start_col, df.shape[1], 2):
        if col + 1 >= df.shape[1]:
            continue

        min_peak_distance = pd.to_numeric(df.iloc[3, col], errors='coerce')
        if not np.isnan(min_peak_distance):
            min_peak_distances[col] = min_peak_distance


    for col in range(channel_start_col, df.shape[1], 2):
        if col + 1 >= df.shape[1]:
            continue

        signal_quality = df.iloc[2, col]
        print(f"Processing Channel Quality: {signal_quality} at column {col}")

        if signal_type == 'extracellular':
            if signal_quality not in ['A', 'B']:
                print(f"Channel in column {col} ({signal_type}) has quality {signal_quality}, skipping.")
                continue
        elif signal_type == 'intracellular':
            if signal_quality != 'A':
                print(f"Channel in column {col} ({signal_type}) has quality {signal_quality}, skipping.")
                continue

        min_peak_distance = pd.to_numeric(df.iloc[3, col], errors='coerce')
        if np.isnan(min_peak_distance):
            if min_peak_distances:
                min_peak_distance = list(min_peak_distances.values())[0]
                print(f"Using min_peak_distance from another channel: {min_peak_distance}")
            else:
                min_peak_distance = default_min_peak_distance
                print(f"Using default min_peak_distance: {min_peak_distance}")

        channel_name = df.iloc[0, col]
        print(f"Processing Channel: {channel_name} ({signal_type})")

        signal_series = pd.to_numeric(df.iloc[10:, col + 1], errors='coerce').dropna().reset_index(drop=True)

        if len(signal_series) == 0:
            print(f"Channel {channel_name} ({signal_type}) has empty signal_series, skipping.")
            continue

        min_peak_distance_point = min_peak_distance * 20000

        segmentation_and_save(signal_series, channel_name, signal_type, signal_dict, excel_prefix, save_dir,
                              min_peak_distance=min_peak_distance_point)


def segmentation_and_save(signal_series, channel_name, signal_type, signal_dict, base_name, plot_dir,
                          min_peak_distance):

    if signal_type == 'extracellular':
        threshold1 = 0.5
        threshold2 = 0.2
        pre_peak_ratio = 0.25
        post_peak_ratio = 0.75
    elif signal_type == 'intracellular':
        threshold1 = 0.3
        threshold2 = 0.2
        pre_peak_ratio = 0.25
        post_peak_ratio = 0.75
    else:
        threshold1 = 0.6
        threshold2 = 0.2
        pre_peak_ratio = 0.25
        post_peak_ratio = 0.75

    min_signal_value = np.min(signal_series)
    max_signal_value = np.max(signal_series)

    if abs(max_signal_value) > abs(min_signal_value):
        print(f"Using peaks for segmentation on {channel_name} ({signal_type})")
        min_peak_height = (max_signal_value - min_signal_value) * threshold1 + min_signal_value
        min_peak_prominence = (max_signal_value - min_signal_value) * threshold2
        peaks, _ = sig.find_peaks(signal_series, height=[min_peak_height, max_signal_value],
                                  prominence=min_peak_prominence, distance=min_peak_distance)
    else:
        # 使用波谷进行分割
        print(f"Using troughs for segmentation on {channel_name} ({signal_type})")
        inverted_signal = -signal_series
        min_peak_height = (max_signal_value - min_signal_value) * threshold1 - max_signal_value
        min_peak_prominence = (max_signal_value - min_signal_value) * threshold2
        peaks, _ = sig.find_peaks(inverted_signal, height=[min_peak_height, -min_signal_value],
                                  prominence=min_peak_prominence, distance=min_peak_distance)

    print(f"Original peaks (or troughs for extracellular) for {channel_name}: {peaks}")

    min_peaks = 2
    if len(peaks) < min_peaks:
        print(f"Channel {channel_name} ({signal_type}) has fewer than {min_peaks} peaks. Skipping.")
        return

    peak_intervals = np.diff(peaks)
    avg_interval = np.mean(peak_intervals)
    print(f"Channel {channel_name} ({signal_type}) Average Peak Interval: {avg_interval:.2f} samples")

    segmented_signals = []
    last_end_idx = 0

    for i, peak in enumerate(peaks):
        start_idx = int(max(last_end_idx, peak - pre_peak_ratio * avg_interval))
        end_idx = int(min(len(signal_series) - 1, peak + post_peak_ratio * avg_interval))

        segmented_signal = signal_series[start_idx:end_idx]
        if len(segmented_signal) < avg_interval * (pre_peak_ratio + post_peak_ratio) * 0.8:
            print(f"Skipping segment {i + 1} of {channel_name} due to insufficient length.")
            continue

        last_end_idx = end_idx

        col_name = f"{base_name}_{channel_name}-Segment_{i + 1}"
        signal_dict[col_name] = segmented_signal

    # 保存分割后的信号，用于后续的匹配和绘图
    signal_dict[channel_name] = segmented_signals

    # 绘制原始信号和分割效果，并保存到文件
    plot_signal_with_segments(signal_series, peaks, avg_interval, pre_peak_ratio, post_peak_ratio, channel_name,
                              signal_type, plot_dir, base_name)


def process_excel_file(file_path, intra_output_csv, extra_output_csv, plot_dir, excel_prefix):
    """
    处理 Excel 文件，提取胞内和胞外信号，舍弃胞内信号的第一个片段后进行匹配并保存。
    """
    intra_df = pd.read_excel(file_path, sheet_name='intracellular')
    extra_df = pd.read_excel(file_path, sheet_name='extracellular')

    all_intra_signals = {}  # 存储所有胞内信号
    all_extra_signals = {}  # 存储所有胞外信号

    # 处理胞内信号
    extract_signals(intra_df, 'intracellular', all_intra_signals, excel_prefix, plot_dir)

    # 处理胞外信号
    extract_signals(extra_df, 'extracellular', all_extra_signals, excel_prefix, plot_dir)

    for channel in all_intra_signals:
        if channel in all_extra_signals:
            intra_segments = all_intra_signals[channel]
            extra_segments = all_extra_signals[channel]

            if len(intra_segments) > 1:  # 如果胞内信号有多个片段，舍弃第一个片段
                intra_segments = intra_segments[1:]
                extra_segments = extra_segments[1:]

                # 进行匹配并保存图像
                plot_intra_and_extra(intra_segments, extra_segments, channel, plot_dir, excel_prefix)

    # 保存到 CSV 文件（去掉第一个片段）
    if all_intra_signals:
        max_len = 0
        for channel, segments in all_intra_signals.items():
            if len(segments) > 1:  # 舍弃第一个片段
                all_intra_signals[channel] = segments[1:]
                max_len = max(max_len, len(all_intra_signals[channel]))

        for key in all_intra_signals:
            all_intra_signals[key] = np.pad(all_intra_signals[key], (0, max_len - len(all_intra_signals[key])),
                                            mode='constant', constant_values=np.nan)

        intra_df_combined = pd.DataFrame(all_intra_signals)
        intra_df_combined.to_csv(intra_output_csv, index=False)
        print(f"Intracellular signals saved to {intra_output_csv}")

    if all_extra_signals:
        max_len = max(len(v) for v in all_extra_signals.values())
        for key in all_extra_signals:
            all_extra_signals[key] = np.pad(all_extra_signals[key], (0, max_len - len(all_extra_signals[key])),
                                            mode='constant', constant_values=np.nan)

        extra_df_combined = pd.DataFrame(all_extra_signals)
        extra_df_combined.to_csv(extra_output_csv, index=False)
        print(f"Extracellular signals saved to {extra_output_csv}")


def process_all_excels_in_folder(folder_path, save_dir, plot_dir):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            base_name = os.path.splitext(file_name)[0]
            intra_output_csv = os.path.join(save_dir, f'{base_name}_intra_signals.csv')
            extra_output_csv = os.path.join(save_dir, f'{base_name}_extra_signals.csv')
            process_excel_file(file_path, intra_output_csv, extra_output_csv, plot_dir, base_name)


if __name__ == '__main__':
    folder_path = './data_test'
    save_dir = './segmentation_out/new data'
    plot_dir = './segplot_out_plus'

    # 自动创建输出和绘图目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created plot directory: {plot_dir}")

    # 开始处理 Excel 文件
    process_all_excels_in_folder(folder_path, save_dir, plot_dir)
