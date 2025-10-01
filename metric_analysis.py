import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime


# 均值滤波器
def mean_filter(dat, parameter):
    n = parameter
    template = np.ones(n) / n
    datf = np.full((n - 1) // 2, dat[0])
    datb = np.full((n - 1) // 2, dat[-1])
    datLong = np.concatenate([datf, dat, datb])
    datLong = np.convolve(datLong, template, mode='same')
    datRes = datLong[n - 1:n - 1 + len(dat)]
    return datRes


# 应用填充
def apply_padding(signal, pad_length, mode='reflect'):
    return np.pad(signal, (pad_length, pad_length), mode=mode)


# Savitzky-Golay 滤波
def savgolay_filter(signal, window_length, polyorder, pad_length=200):
    padded_signal = apply_padding(signal, pad_length)
    filtered_padded_signal = savgol_filter(padded_signal, window_length, polyorder)

    return filtered_padded_signal[pad_length:-pad_length]


def calculate_t_and_slope(signal, fs):
    # 滤波
    filtered_signal = savgol_filter(signal, 501, 3)
    max_index_origin = np.argmax(signal)
    min_index_origin = np.argmin(signal)

    # 基线计算（前5%和后5%的平均值）
    baseline = (np.mean(signal[:int(len(signal) * 0.05)]) +
                np.mean(signal[-int(len(signal) * 0.05):])) / 2

    # 找滤波信号的最大值和最小值
    max_idx, max_val = np.argmax(filtered_signal), np.max(filtered_signal)
    min_idx, min_val = np.argmin(filtered_signal), np.min(filtered_signal)
    displacement = min_index_origin - min_idx

    # 初始化返回值
    t1, k1, t2, k2 = None, None, None, None
    peak, valley, vmax = None, None, None

    # 如果最大值在最小值右边，说明 T 波更高
    if max_idx > min_idx:
        # 找负峰左侧的最高点作为实际的正峰
        peak_idx = np.argmax(filtered_signal[:min_idx]) + displacement
        peak = signal[peak_idx]  # 在原始信号中标定
        valley_idx = min_idx + displacement
        valley = signal[valley_idx]
        vmax_idx = max_idx + displacement
        vmax = signal[vmax_idx]
    else:
        # 正常情况
        peak_idx = max_idx + displacement
        peak = signal[peak_idx]
        valley_idx = min_idx + displacement
        valley = signal[valley_idx]
        vmax_idx = np.argmax(filtered_signal[min_idx:]) + min_idx + displacement
        vmax = signal[vmax_idx]

    # 计算 t1 和 k1
    start_idx = np.where(filtered_signal[:peak_idx] <= baseline)[0]
    if len(start_idx) > 0:
        start_idx = start_idx[-1]
        t1 = (peak_idx - start_idx) / fs
        k1 = (signal[peak_idx] - signal[start_idx]) / (peak_idx - start_idx)

    # 计算 t2 和 k2
    end_idx = np.where(filtered_signal[vmax_idx:] <= baseline)[0]
    if len(end_idx) > 0:
        end_idx = vmax_idx + end_idx[0]
        t2 = (end_idx - min_idx) / fs
        k2 = (signal[end_idx] - signal[min_idx]) / (end_idx - min_idx)

    # 返回计算结果
    return t1, k1, t2, k2, peak, valley, vmax, peak_idx, min_idx, vmax_idx


def find_pacing_points_and_apd(segmented_signals, fs):
    """
    计算 APD10 到 APD100，以及去极化时间和复极化时间。
    参数:
        segmented_signals: 输入的信号片段
        fs: 采样频率
        ap_up_wind_len: 起搏窗口长度的比例
    返回:
        apd_values: 包含 APD10 到 APD100 的字典
        depolarization_time: 去极化时间
        repolarization_time: 复极化时间
    """
    apd_values = {f"APD{i}": None for i in range(10, 101, 10)}
    depolarization_time = None
    repolarization_time = None
    ap_up_determine = 0
    mean_window = 0
    std_window = 0

    # 将信号转换为 1D 数组
    signal_np = np.array(segmented_signals)
    max_index_origin = np.argmax(signal_np)
    print(signal_np.shape)
    if signal_np.ndim > 1:
        signal_np = signal_np.flatten()

    # 信号平滑处理
    filtered_signal = mean_filter(signal_np, 50)
    filtered_signal = mean_filter(filtered_signal, 50)
    filtered_signal = mean_filter(filtered_signal, 50)

    # 检测最大值及其位置（基于滤波信号）
    max_index = np.argmax(filtered_signal)
    max_value = signal_np[max_index]  # 原始信号的峰值
    displacement = max_index_origin - max_index

    # 起搏点检测（基于滤波信号）
    start_idx = max(0, max_index - 1000)

    candidate_start = None
    baseline_value = np.mean(filtered_signal[max_index-800:max_index - 400])
    print('baseline_value:', baseline_value)
    num_base = int(max_index * 0.7)
    baseline = np.ones(num_base) * baseline_value

    for idx in range(start_idx, max_index):
        process_point = filtered_signal[idx]
        window_data = np.append(filtered_signal[0:idx], baseline)

        # 动态调整检测阈值
        mean_window = np.mean(window_data)
        std_window = np.std(window_data)

        if std_window < 0.001:
            threshold = calculate_ap_up_determine(std_window)
            std_window = std_window * threshold

        # 动态计算 ap_up_determine
        ap_up_detect = 5
        threshold1 = mean_window + ap_up_detect * std_window
        if threshold1 >= 0.6:
            threshold1 = 0.6

        if std_window < 0.02:
            ap_up_determine = 3
        elif 0.02 < std_window < 0.04:
            ap_up_determine = 1
        elif 0.056 < std_window < 0.07:
            ap_up_determine = -0.3
        elif 0.07 < std_window < 0.08:
            ap_up_determine = -0.15
        elif 0.08 < std_window < 0.1:
            ap_up_determine = -0.25
        elif 0.1 < std_window < 0.12:
            ap_up_determine = -0.4
        elif 0.12 < std_window < 0.15:
            ap_up_determine = -0.6
        elif 0.15 < std_window:
            ap_up_determine = -0.9
        else:
            ap_up_determine = 0.6

        if process_point > threshold1:
            candidate_start = idx
            break

    print('平均值为:', mean_window)
    print('标准差为:', std_window)

    start_point = None
    if candidate_start is not None:
        threshold2 = mean_window + ap_up_determine * std_window
        print('threshold：', threshold2)
        for idx in range(candidate_start, 0, -1):
            if filtered_signal[idx] < threshold2:
                start_point = idx
                break

    if start_point is None:
        if start_idx < max_index:
            sub_signal = filtered_signal[start_idx:max_index]
            start_point = np.argmin(sub_signal) + start_idx if len(sub_signal) > 0 else 0
        else:
            start_point = 0

    start_point = start_point + displacement

    # 计算去极化时间
    if start_point is not None:
        depolarization_time = (max_index - start_point) / fs

        # 检测 APD10 到 APD100（基于原始信号）
        for percentage in range(10, 101, 10):
            threshold_left = signal_np[start_point] + ((100 - percentage) / 100) * (max_value - signal_np[start_point])
            left_idx = np.where(signal_np[:max_index] <= threshold_left)[0]
            valley_index = np.argmin(signal_np[max_index:]) + max_index  # 波谷点索引
            threshold_right = signal_np[valley_index] + ((100 - percentage) / 100) * (max_value - signal_np[valley_index])
            right_idx = np.where(signal_np[max_index:] <= threshold_right)[0]

            if len(left_idx) > 0 and len(right_idx) > 0:
                apd_duration = (right_idx[0] + max_index) - left_idx[-1]
                apd_values[f"APD{percentage}"] = apd_duration / fs

        # 检测复极化时间（基于原始信号）
        min_index_after_max = np.argmin(signal_np[max_index:]) + max_index
        if min_index_after_max > max_index:
            repolarization_time = (min_index_after_max - max_index) / fs

    return apd_values, depolarization_time, repolarization_time, filtered_signal, start_point, baseline_value


def calculate_ap_up_determine(std_window):
    """
    根据 std_window 动态调整 ap_up_determine 的值
    """
    if std_window <= 0:
        std_window = 1e-16

    # 使用对数计算数量级
    magnitude = -np.log10(std_window) - 1  # 标准差的数量级

    ap_up_determine = 10 ** int(np.floor(magnitude))  # 每降低一个数量级，放大 10 倍

    return ap_up_determine


# 动态创建路径和文件夹
def create_output_directory(base_dir, input_file):
    # 创建主输出目录
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 提取输入文件名（不带扩展名）并添加时间戳
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = f"{file_name}_{timestamp}"

    # 创建子目录
    full_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(full_dir, exist_ok=True)
    return full_dir


# 读取 Excel 文件
def read_data_from_excel(file_path):
    excel_data = pd.ExcelFile(file_path)
    all_predictions = []
    all_targets = []
    all_extras = []
    sample_indices = []

    # 假设预测、真实值和额外信号分别存储在三个不同的 sheet 中
    predictions_sheet = excel_data.parse("Predictions")
    targets_sheet = excel_data.parse("True Values")
    extras_sheet = excel_data.parse("Extra")

    sample_indices = [f"Sample_{i}" for i in range(len(predictions_sheet))]  # 样本索引

    # 获取每列的数据（按列读取）
    all_predictions = predictions_sheet.T.values  # 转置操作，将行数据转换为列数据
    all_targets = targets_sheet.T.values  # 转置操作
    all_extras = extras_sheet.T.values  # 转置操作

    return all_predictions, all_targets, all_extras, sample_indices


# 修改胞内信号处理函数，传递输出目录
def process_intracellular_signals(predictions_list, targets_list, sample_indices, fs, output_dir):
    results = []

    for i, (predictions, targets) in enumerate(
            tqdm(zip(predictions_list, targets_list), desc="Processing Intracellular Signals")):
        # 计算 APD 和复极化时间，同时获取平滑后的信号和起搏点
        target_apd, target_depol, target_repol, filter_target, pacing_point_target, base_value_target = find_pacing_points_and_apd(
            targets, fs)
        (prediction_apd, prediction_depol, prediction_repol, filter_prediction, pacing_point_prediction,
         base_value_prediction) = find_pacing_points_and_apd(predictions, fs)

        # 计算 R2 和 MSE
        r2 = r2_score(targets, predictions)
        mse = mean_squared_error(targets, predictions)

        # 整理结果
        row = [sample_indices[i]]  # 样本索引
        row += [target_apd[f"APD{j}"] for j in range(10, 101, 10)]  # APD_Target
        row += [prediction_apd[f"APD{j}"] for j in range(10, 101, 10)]  # APD_Prediction
        row += [target_depol, prediction_depol, target_repol, prediction_repol, r2, mse]

        results.append(row)

    return results


# 保存多 Sheet 的 Excel 文件
def save_results_to_excel(intra_results, output_file):
    with pd.ExcelWriter(output_file) as writer:
        # 保存胞内信号结果
        intra_columns = (
                ["Sample Index"]
                + [f"APD{i}_Target" for i in range(10, 101, 10)]
                + [f"APD{i}_Prediction" for i in range(10, 101, 10)]
                + [
                    "Depolarization Time_Target",
                    "Depolarization Time_Prediction",
                    "Repolarization Time_Target",
                    "Repolarization Time_Prediction",
                    "R2",
                    "MSE",
                ]
        )
        intra_df = pd.DataFrame(intra_results, columns=intra_columns)
        intra_df.to_excel(writer, index=False, sheet_name="Intra")

    print(f"结果已保存至 {output_file}")


# 主程序，处理文件夹中的所有 Excel 文件
def main(input_folder='./apd_plot'):
    fs = 20000  # 假设采样率

    # 遍历文件夹中的所有 Excel 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('_predictions_and_targets.xlsx'):
            file_path = os.path.join(input_folder, file_name)
            base_name = file_name.replace('_predictions_and_targets.xlsx', '')

            # 创建输出文件夹
            output_dir = create_output_directory(input_folder, file_path)

            # 处理 Excel 文件中的信号
            predictions, targets, extras, sample_indices = read_data_from_excel(file_path)
            intra_results = process_intracellular_signals(predictions, targets, sample_indices, fs, output_dir)

            # 保存结果到新的 Excel 文件
            output_file = os.path.join(output_dir, f"{base_name}_auto_analysis_results.xlsx")
            save_results_to_excel(intra_results, output_file)

            print(f"处理完成: {file_name}")


if __name__ == "__main__":
    main()
