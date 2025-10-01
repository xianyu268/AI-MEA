import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
import ctypes
import gc
from tqdm import tqdm
from datetime import datetime
from torchinfo import summary
from scipy.interpolate import interp1d
from torch.nn import Module

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


# 获取短路径函数
def get_short_path_name(long_name):
    buf_size = ctypes.windll.kernel32.GetShortPathNameW(long_name, None, 0)
    buffer = ctypes.create_unicode_buffer(buf_size)
    ctypes.windll.kernel32.GetShortPathNameW(long_name, buffer, buf_size)
    return buffer.value


# 加载数据
# 从 npy 文件加载原始数据
raw_y_data, raw_x_data = np.load('./npy_file/train_data_R5V7.npy')
raw_y_test, raw_x_test = np.load('./npy_file/test_data_R5V7.npy')
print(raw_y_data.shape, raw_x_data.shape)


def downsample_data(data, original_length=10000, target_length=6000):
    """
    将数据从 original_length 降采样到 target_length。
    :param data: 输入数据，形状为 (num_samples, original_length)
    :return: 降采样后的数据，形状为 (num_samples, target_length)
    """
    x_original = np.linspace(0, original_length - 1, original_length)
    x_target = np.linspace(0, original_length - 1, target_length)
    downsampled_data = np.array([
        interp1d(x_original, sample, kind='linear', fill_value="extrapolate")(x_target)
        for sample in data
    ])
    return downsampled_data


def upsample_data(data, original_length=6000, target_length=10000):
    """
    将数据从 original_length 上采样到 target_length。
    :param data: 输入数据，形状为 (num_samples, original_length)
    :return: 上采样后的数据，形状为 (num_samples, target_length)
    """
    x_original = np.linspace(0, original_length - 1, original_length)
    x_target = np.linspace(0, original_length - 1, target_length)
    upsampled_data = np.array([
        interp1d(x_original, sample, kind='linear', fill_value="extrapolate")(x_target)
        for sample in data
    ])
    return upsampled_data


# 数据归一化
def normalize_data(data):
    """
    对二维数据逐列归一化
    :param data: 输入数据，形状为 (num_samples, num_features)
    :return: 归一化后的数据，形状不变
    """
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        column = data[:, i]
        _range = np.max(column) - np.min(column)
        normalized_data[:, i] = (column - np.min(column)) / (_range + 1e-8)  # 避免除零
    return normalized_data

# 归一化数据
raw_x_data = normalize_data(raw_x_data)
raw_y_data = normalize_data(raw_y_data)
raw_x_test = normalize_data(raw_x_test)
raw_y_test = normalize_data(raw_y_test)

raw_y_test = raw_y_test.T
raw_x_test = raw_x_test.T
raw_y_data = raw_y_data.T
raw_x_data = raw_x_data.T

# 数据降采样
original_length = 10000
target_length = 6000

# 数据降采样
x_data = downsample_data(raw_x_data, original_length, target_length)
y_data = downsample_data(raw_y_data, original_length, target_length)
x_test = downsample_data(raw_x_test, original_length, target_length)
y_test = downsample_data(raw_y_test, original_length, target_length)

# 划分训练和验证集
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=54)

# 转换数据为PyTorch张量并调整形状
train_X = torch.tensor(x_train, dtype=torch.float32).unsqueeze(2)
train_Y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(2)
val_X = torch.tensor(x_val, dtype=torch.float32).unsqueeze(2)
val_Y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(2)
test_X = torch.tensor(x_test, dtype=torch.float32).unsqueeze(2)
test_Y = torch.tensor(y_test, dtype=torch.float32).unsqueeze(2)

# 创建数据加载器
train_dataset = TensorDataset(train_X, train_Y)
val_dataset = TensorDataset(val_X, val_Y)
test_dataset = TensorDataset(test_X, test_Y)

train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义多个卷积层和池化层
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=10, padding=5)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, padding=5)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv1d_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=5)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv1d_4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=10, padding=5)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.dropout4 = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)

        # 反卷积层，用于上采样
        self.deconv1d_4 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.deconv1d_1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.deconv1d_2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv1d_3 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=2, stride=2)

        """
        para conv[1,32,64]   deconv[128,64,32]
        """

    def forward(self, x):
        x = x.transpose(1, 2)  # 调整形状以适应Conv1d的输入 (batch_size, in_channels, seq_length)

        x = F.relu(self.conv1d_1(x))
        # x = self.batchnorm1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv1d_2(x))
        # x = self.batchnorm2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv1d_3(x))
        # x = self.batchnorm3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # x = F.relu(self.conv1d_4(x))
        # # x = self.batchnorm4(x)
        # x = self.pool4(x)
        # x = self.dropout4(x)

        x = x.transpose(1, 2)  # 调整回LSTM的输入 (batch_size, seq_length, out_channels)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).detach()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).detach()

        encoder_outputs, (hn, cn) = self.lstm(x, (h0, c0))

        outputs = self.fc(encoder_outputs)

        # 上采样
        outputs = outputs.transpose(1, 2)  # 调整形状以适应ConvTranspose1d的输入 (batch_size, out_channels, seq_length)
        # outputs = F.relu(self.deconv1d_4(outputs))
        outputs = F.relu(self.deconv1d_1(outputs))
        outputs = F.relu(self.deconv1d_2(outputs))
        outputs = F.relu(self.deconv1d_3(outputs))
        outputs = outputs.transpose(1, 2)  # 调整回原始形状 (batch_size, seq_length, 1)

        return outputs


def mean_filter(dat, parameter):
    n = parameter
    template = np.ones(n) / n
    datf = np.full((n - 1) // 2, dat[0])
    datb = np.full((n - 1) // 2, dat[-1])
    datLong = np.concatenate([datf, dat, datb])
    datLong = np.convolve(datLong, template, mode='same')
    datRes = datLong[n - 1:n - 1 + len(dat)]

    return datRes


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


def find_pacing_points_and_apd50(segmented_signals, fs):
    pacing_point = []
    apd50_value = []
    extra_apd_value = []
    repolarization_time = []
    depolarization_time = []
    left_points = []
    right_points = []
    ap_up_determine = 0
    mean_window = 0
    std_window = 0

    signal_np = np.array(segmented_signals)
    if signal_np.ndim == 3:
        signal_np = signal_np[0, :, 0]
    elif signal_np.ndim == 2:
        signal_np = signal_np[0]
    # print(signal_np.shape)

    filtered_signal = mean_filter(signal_np, 5)
    filtered_signal = mean_filter(filtered_signal, 11)
    filtered_signal = mean_filter(filtered_signal, 15)

    # 检测最大值及其位置（基于滤波信号）
    max_index = np.argmax(filtered_signal)
    max_index_origin = np.argmax(signal_np)
    max_value = signal_np[max_index]  # 原始信号的峰值
    displacement = max_index_origin - max_index

    # 起搏点检测（基于滤波信号）
    start_idx = max(0, max_index - 1000)

    candidate_start = None
    baseline_value = np.mean(filtered_signal[max_index - 800:max_index - 400])
    # print('baseline_value:', baseline_value)
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

    # print('平均值为:', mean_window)
    # print('标准差为:', std_window)

    start_point = None
    if candidate_start is not None:
        threshold2 = mean_window + ap_up_determine * std_window
        # print('threshold：', threshold2)
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

    # 起搏点检测成功，执行后续逻辑
    if start_point is not None:
        half_max_value = (max_value + signal_np[start_point]) / 2
        left_idx = np.where(signal_np[:max_index] <= half_max_value)[0]
        right_idx = np.where(signal_np[max_index:] <= half_max_value)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            apd50 = (right_idx[0] + max_index) - left_idx[-1]
            pacing_point.append(start_point)
            apd50_value.append(apd50 / fs)

            # 查找最大值后的最小值
            min_index_after_max = np.argmin(filtered_signal[max_index:]) + max_index

            # 计算额外 APD
            extra_apd_start = signal_np[start_point]
            extra_apd_idx = np.where(signal_np[max_index:] == extra_apd_start)[0]
            if len(extra_apd_idx) > 0:
                extra_apd = (extra_apd_idx[0] + max_index - start_point) / fs
                extra_apd_value.append(extra_apd )
            else:
                extra_apd = (min_index_after_max - start_point) /fs
                extra_apd_value.append(extra_apd)

            depolarization_duration = (max_index_origin - start_point) / fs
            depolarization_time.append(depolarization_duration)

            # 计算复极化时间
            if min_index_after_max > max_index:
                repolarization_duration = (min_index_after_max - max_index) / fs
                repolarization_time.append(repolarization_duration)

            left_points.append(left_idx[-1])
            right_points.append(right_idx[0] + max_index)

        # print(f'找到起搏点，索引：{start_point}, 值：{filtered_signal[start_point]}')
        # print(f'最大值索引：{max_index}, 值：{max_value}')
        # # 打印APD50值和复极化时间
        # print(f'APD50: {apd50_value}, APD: {extra_apd_value}, 复极化时间: {repolarization_time}')

    return pacing_point, apd50_value, depolarization_time, repolarization_time, left_points, right_points


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        # 找到预测输出和真实值的APD50
        pacing_points_output, apd50_output, depolarization_output, repolarization_output, _, _ = find_pacing_points_and_apd50(
            output.detach().cpu().numpy(), fs=20000)
        pacing_points_target, apd50_target, depolarization_target, repolarization_target, _, _ = find_pacing_points_and_apd50(
            target.detach().cpu().numpy(), fs=20000)

        # 检查是否有有效的APD50或起搏点
        max_index = np.argmax(output.detach().cpu().numpy())

        if max_index == 0 or len(apd50_output) == 0 or len(apd50_target) == 0 or len(depolarization_output) == 0 or len(repolarization_output) == 0:
            # 如果模型输出接近直线（max_index == 0），仅使用SmoothL1Loss
            # print("输出仍为直线，仅使用SmoothL1Loss")
            mse_loss = F.mse_loss(output, target)
            return mse_loss

        apd50_output_tensor = torch.from_numpy(np.array(apd50_output)).float().to(output.device)
        apd50_target_tensor = torch.from_numpy(np.array(apd50_target)).float().to(output.device)

        depolarization_output_tensor = torch.from_numpy(np.array(depolarization_output)).float().to(output.device)
        depolarization_target_tensor = torch.from_numpy(np.array(depolarization_target)).float().to(output.device)

        repolarization_output_tensor = torch.from_numpy(np.array(repolarization_output)).float().to(output.device)
        repolarization_target_tensor = torch.from_numpy(np.array(repolarization_target)).float().to(output.device)

        # 计算基础的smooth_l1_loss损失
        mse_loss = F.mse_loss(output, target)

        # 计算APD50损失
        apd50_loss = F.mse_loss(apd50_output_tensor, apd50_target_tensor)

        # 计算去极化时间损失
        depolarization_loss = F.mse_loss(depolarization_output_tensor, depolarization_target_tensor)

        # 计算复极化时间损失
        repolarization_loss = F.mse_loss(repolarization_output_tensor, repolarization_target_tensor)

        return mse_loss + 0.01 * depolarization_loss + 2 * apd50_loss + 0.01 * repolarization_loss


# 初始化模型参数
input_size = 1
hidden_size = 128
num_layers = 1
output_size = 1

model = CNNLSTM(input_size, hidden_size, num_layers, output_size).to(device)

# 定义损失函数和优化器
# criterion = nn.MSELoss()
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 创建 TensorBoard 记录器
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'./runs/Seq2SeqLSTM_designloss_{current_time}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_writer = SummaryWriter(log_dir=log_dir)

# 模型保存路径
model_save_path = os.path.join(log_dir, 'best_model.pth')


class ModelWrapper(Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 确保输入形状与模型要求一致
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 增加一个通道维度
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            return outputs[0]  # 返回第一个输出
        return outputs


def save_model_parameters(model, file_path):
    """
    保存模型的所有参数（包括层的名称、维度和超参数）到文本文件中。
    :param model: PyTorch 模型实例
    :param file_path: 保存文件的路径
    """
    with open(file_path, 'w') as f:
        f.write("Model Summary\n")
        f.write("=" * 50 + "\n\n")

        # 捕获 torchinfo 的输出
        summary_output = summary(
            model,
            input_size=(1, train_X.size(-1)),  # 确保输入维度正确
            device=str(device),
            col_names=["output_size", "num_params"],
            verbose=0  # 控制台输出关闭
        )

        # 将模型摘要写入文件
        f.write(str(summary_output))

        f.write("\n\nDetailed Parameters\n")
        f.write("=" * 50 + "\n")
        for name, param in model.named_parameters():
            f.write(f"Layer: {name}\n")
            f.write(f"  Shape: {param.size()}\n")
            f.write(f"  Requires Grad: {param.requires_grad}\n")
            f.write("\n")

    print(f"Model parameters saved to {file_path}")


# 训练逻辑
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_path, results_dir,
                patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0  # 用于记录验证集损失未改善的连续周期数
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        model.train()
        train_losses = []
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                pbar.update(1)

        avg_train_loss = sum(train_losses) / len(train_losses)
        tensorboard_writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # Validation loss
        model.eval()
        val_losses = []
        with tqdm(total=len(val_loader), desc='Validation', unit='batch') as pbar:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    val_losses.append(val_loss.item())
                    pbar.update(1)

        avg_val_loss = sum(val_losses) / len(val_losses)
        tensorboard_writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

        # 检查是否为最佳验证损失
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0  # 重置未改善周期计数器
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        # 检查是否需要早停
        if epochs_no_improve >= patience:
            print("Early stopping criteria met. Stopping training.")
            early_stop = True

    # 关闭 TensorBoard 记录器
    tensorboard_writer.close()


# 测试逻辑
def test_model(model, test_loader, model_path, results_dir, raw_x_test, raw_y_test, original_length=10000,
               target_length=6000):
    """
    测试模型，并将预测值、真实值和额外的输入信号保存到一个 Excel 文件的不同 sheets 中。
    :param model: 训练好的模型
    :param test_loader: 测试数据加载器（降采样后的数据）
    :param model_path: 训练好的模型路径
    :param results_dir: 保存结果的目录
    :param raw_x_test: 原始 10000 点的胞外信号
    :param raw_y_test: 原始 10000 点的胞内信号
    :param original_length: 原始信号长度
    :param target_length: 降采样信号长度
    """
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_losses = []

    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)

    # 创建一个新的 Excel 文件
    excel_file_path = os.path.join(results_dir, 'predictions_and_targets.xlsx')

    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        predictions_list = []  # 用于存储所有预测值
        true_values_list = []  # 用于存储所有真实值
        extra_list = []  # 用于存储额外的输入信号

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss = criterion(outputs, targets)
                    test_losses.append(test_loss.item())

                    # 将模型输出从 6000 点上采样到 10000 点
                    outputs_np = outputs.cpu().numpy().squeeze(2)
                    outputs_upsampled = upsample_data(outputs_np, target_length, original_length)

                    # 获取当前批次的原始信号
                    batch_start = i * test_loader.batch_size
                    batch_end = batch_start + inputs.size(0)

                    raw_inputs_batch = raw_x_test[batch_start:batch_end]
                    raw_targets_batch = raw_y_test[batch_start:batch_end]

                    # 将当前批次的结果添加到各自的列表中
                    for j in range(inputs.size(0)):  # 遍历 batch 内的每个样本
                        predictions_list.append(outputs_upsampled[j])
                        true_values_list.append(raw_targets_batch[j])
                        extra_list.append(raw_inputs_batch[j])

                    pbar.update(1)

        # 将每个列表的内容转换为 DataFrame，并保存到不同的 sheet 中
        predictions_df = pd.DataFrame(predictions_list)
        true_values_df = pd.DataFrame(true_values_list)
        extra_df = pd.DataFrame(extra_list)

        # 为 DataFrame 设置列名
        sample_names = [f'Sample_{i}' for i in range(len(predictions_list))]
        predictions_df.columns = sample_names
        true_values_df.columns = sample_names
        extra_df.columns = sample_names

        # 将 DataFrame 写入 Excel 的不同 sheet
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        true_values_df.to_excel(writer, sheet_name='True Values', index=False)
        extra_df.to_excel(writer, sheet_name='Extra', index=False)

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f'Average Test Loss: {avg_test_loss}')
    print(f'Results saved to {excel_file_path}')
    return avg_test_loss


# 控制模式
MODE = "train+predict"  # 可以是 "train", "predict", 或 "train+predict"


# 模型参数保存
model_summary_path = os.path.join(log_dir, 'model_parameters.txt')

wrapped_model = ModelWrapper(model)
# print(summary(wrapped_model, input_size=(1, train_X.size(-1)), device=str(device)))
save_model_parameters(wrapped_model, model_summary_path)

# 训练和测试执行
if MODE in ["train", "train+predict"]:
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=1000,
                model_path=model_save_path, results_dir=log_dir, patience=20)
    print("Training completed.")

if MODE in ["predict", "train+predict"]:
    print("Starting testing...")
    test_loss = test_model(
        model=model,
        test_loader=test_loader,
        model_path=model_save_path,
        results_dir=log_dir,
        raw_x_test=raw_x_test,
        raw_y_test=raw_y_test,
        original_length=10000,
        target_length=6000
    )
    print(f"Testing completed. Test Loss: {test_loss:.4f}")
