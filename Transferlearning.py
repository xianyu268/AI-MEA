import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import interp1d

## 数据集路径
train_data_path = './npy_file/training_dataset_1000.npy'
test_data_path = './npy_file/testing_dataset_500.npy'

raw_y_data, raw_x_data = np.load(train_data_path)
raw_y_test, raw_x_test = np.load(test_data_path)
print(f"训练数据形状: {raw_y_data.shape, raw_x_data.shape}")
print(f"测试数据形状: {raw_y_test.shape, raw_x_test.shape}")


def downsample_data(data, original_length=10000, target_length=6000):
    x_original = np.linspace(0, original_length - 1, original_length)
    x_target = np.linspace(0, original_length - 1, target_length)
    downsampled_data = np.array([
        interp1d(x_original, sample, kind='linear', fill_value="extrapolate")(x_target)
        for sample in data
    ])
    return downsampled_data


def upsample_data(data, original_length=6000, target_length=10000):
    x_original = np.linspace(0, original_length - 1, original_length)
    x_target = np.linspace(0, original_length - 1, target_length)
    upsampled_data = np.array([
        interp1d(x_original, sample, kind='linear', fill_value="extrapolate")(x_target)
        for sample in data
    ])
    return upsampled_data


def normalize_data(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        column = data[:, i]
        _range = np.max(column) - np.min(column)
        normalized_data[:, i] = (column - np.min(column)) / (_range + 1e-8)
    return normalized_data

raw_x_data = normalize_data(raw_x_data)
raw_y_data = normalize_data(raw_y_data)
raw_x_test = normalize_data(raw_x_test)
raw_y_test = normalize_data(raw_y_test)

raw_y_test = raw_y_test.T
raw_x_test = raw_x_test.T
raw_y_data = raw_y_data.T
raw_x_data = raw_x_data.T

original_length = 10000
target_length = 6000

x_data = downsample_data(raw_x_data, original_length, target_length)
y_data = downsample_data(raw_y_data, original_length, target_length)
x_test = downsample_data(raw_x_test, original_length, target_length)
y_test = downsample_data(raw_y_test, original_length, target_length)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=70)

train_X = torch.tensor(x_train, dtype=torch.float32).unsqueeze(2)
train_Y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(2)
val_X = torch.tensor(x_val, dtype=torch.float32).unsqueeze(2)
val_Y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(2)
test_X = torch.tensor(x_test, dtype=torch.float32).unsqueeze(2)
test_Y = torch.tensor(y_test, dtype=torch.float32).unsqueeze(2)

batch_size = 2
train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=batch_size, shuffle=False)

# print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


def freeze_layers(model, layers_to_freeze):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False
            print(f"冻结层: {name}")
        else:
            param.requires_grad = True
            print(f"训练层: {name}")
    return model


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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

        self.deconv1d_4 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.deconv1d_1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.deconv1d_2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv1d_3 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=2, stride=2)

        """
        para conv[1,32,64]   deconv[128,64,32]
        """

    def forward(self, x):
        x = x.transpose(1, 2)

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

        x = x.transpose(1, 2)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).detach()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).detach()

        encoder_outputs, (hn, cn) = self.lstm(x, (h0, c0))

        outputs = self.fc(encoder_outputs)

        outputs = outputs.transpose(1, 2)
        # outputs = F.relu(self.deconv1d_4(outputs))
        outputs = F.relu(self.deconv1d_1(outputs))
        outputs = F.relu(self.deconv1d_2(outputs))
        outputs = F.relu(self.deconv1d_3(outputs))
        outputs = outputs.transpose(1, 2)

        return outputs


def transfer_learning(model, train_loader, val_loader, criterion, optimizer, model_path, epochs=50,
                      layers_to_freeze=[], patience=10):
    """
    迁移学习训练函数，包含早停机制。
    :param model: 要训练的模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param model_path: 保存最佳模型的路径
    :param epochs: 最大训练轮数
    :param layers_to_freeze: 需要冻结的层列表
    :param patience: 早停机制的耐心值，当验证集损失在指定轮次内不下降时停止训练
    """
    model = freeze_layers(model, layers_to_freeze)
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"验证集损失未改善 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"验证集损失在 {patience} 个 epoch 内未改善，提前停止训练。")
                break

    print(f"训练完成。最佳模型出现在第 {best_epoch+1} 轮，验证损失: {best_val_loss:.4f}")


def test_model(model, test_loader, model_path, results_dir, raw_x_test, raw_y_test, original_length=10000,
               target_length=6000):

    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_losses = []

    os.makedirs(results_dir, exist_ok=True)

    excel_file_path = os.path.join(results_dir, 'predictions_and_targets.xlsx')

    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        predictions_list = []
        true_values_list = []
        extra_list = []

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss = criterion(outputs, targets)
                    test_losses.append(test_loss.item())

                    outputs_np = outputs.cpu().numpy().squeeze(2)
                    outputs_upsampled = upsample_data(outputs_np, target_length, original_length)

                    batch_start = i * test_loader.batch_size
                    batch_end = batch_start + inputs.size(0)

                    raw_inputs_batch = raw_x_test[batch_start:batch_end]
                    raw_targets_batch = raw_y_test[batch_start:batch_end]

                    for j in range(inputs.size(0)):
                        predictions_list.append(outputs_upsampled[j])
                        true_values_list.append(raw_targets_batch[j])
                        extra_list.append(raw_inputs_batch[j])

                    pbar.update(1)

        predictions_df = pd.DataFrame(predictions_list)
        true_values_df = pd.DataFrame(true_values_list)
        extra_df = pd.DataFrame(extra_list)

        sample_names = [f'Sample_{i}' for i in range(len(predictions_list))]
        predictions_df.columns = sample_names
        true_values_df.columns = sample_names
        extra_df.columns = sample_names

        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        true_values_df.to_excel(writer, sheet_name='True Values', index=False)
        extra_df.to_excel(writer, sheet_name='Extra', index=False)

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f'Average Test Loss: {avg_test_loss}')
    print(f'Results saved to {excel_file_path}')
    return avg_test_loss


LAYERS_TO_FREEZE = ['conv1d_1', "pool1", "dropout1", 'conv1d_2', "pool2", "dropout2", 'conv1d_3', "pool3", "dropout3"]
input_size, hidden_size, num_layers, output_size = 1, 128, 1, 1
model = CNNLSTM(input_size, hidden_size, num_layers, output_size).to(device)

pretrained_model_path = './runs/Seq2SeqLSTM_R5V7_BEST/best_model.pth'
if os.path.exists(pretrained_model_path):
    print("加载预训练模型权重...")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print("预训练模型加载成功！")
else:
    print("未找到预训练模型，训练将从头开始！")

criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = './ZCH-save/results/3_Predict'  ##保存目录
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model_save_path = os.path.join(log_dir, 'transferlearning_xxy_20250731.pth')   ## 记得改名字注意不要覆盖了

TRANSFER_MODE = "train"

if TRANSFER_MODE in ["train", "train+test"]:
    print("开始迁移学习训练...")
    transfer_learning(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_save_path,
        epochs=500,  ##训练轮次
        layers_to_freeze=LAYERS_TO_FREEZE,
        patience=30   ##多少个轮次如果没改善就暂停
    )


if TRANSFER_MODE in ["test", "train+test"]:
    print("加载最佳模型进行测试...")
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
