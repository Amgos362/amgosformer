import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt

# rolling minmax scaling 함수 (window=24)
def rolling_minmax_scale(series, window=24):
    roll_min = series.rolling(window=window, min_periods=window).min()
    roll_max = series.rolling(window=window, min_periods=window).max()
    scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-8)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    scaled = scaled.fillna(1.0)
    return scaled.clip(upper=1.0)

# binning 및 one-hot 인코딩 함수 (결과를 정수 0,1로)
def bin_and_encode(data, features, bins=100, drop_original=True):
    for feature in features:
        # pd.cut에 bins로 지정된 구간을 사용 (여기서는 자동으로 최소~최대 값)
        data[f'{feature}_Bin'] = pd.cut(data[feature], bins=bins, labels=False)
        one_hot = pd.get_dummies(data[f'{feature}_Bin'], prefix=f'{feature}_Bin').astype(np.int32)
        expected_columns = [f'{feature}_Bin_{i}' for i in range(bins)]
        one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
        data = pd.concat([data, one_hot], axis=1)
        if drop_original:
            data.drop(columns=[f'{feature}_Bin'], inplace=True)
    # 개별 열 단위로 float32로 변환
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        data[col] = data[col].astype(np.float32)
    return data

####################################
# Dataset 및 모델 정의 (나머지 구성은 동일)
####################################
class TimeSeriesDataset(Dataset):
    def __init__(self, input_data, target_data, lookback=24):
        self.input_data = input_data.values
        self.target_data = target_data.values
        self.lookback = lookback

    def __len__(self):
        return len(self.input_data) - self.lookback

    def __getitem__(self, idx):
        x = self.input_data[idx: idx + self.lookback, :]
        y = self.target_data[idx + self.lookback, 0]
        y_prev = self.target_data[idx + self.lookback - 1, 0]
        y_target = 1 if y > y_prev else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_target, dtype=torch.long)

class EncoderOnlyTransformerCustom(nn.Module):
    def __init__(self, input_dim, embedding_dim=512, num_layers=6, nhead=8, ffn_dim=2048, num_classes=2, max_seq_len=24):
        super(EncoderOnlyTransformerCustom, self).__init__()
        self.token_embedding = nn.Linear(input_dim, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=ffn_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        return self.fc(x[-1, :, :])

####################################
# 데이터 로드 및 Variant 1 실행: OHLC 원본 데이터로 실험
####################################
# CSV 파일 로드 – OHLC 4개 데이터만 사용
data = pd.read_csv("ETH_upbit_KRW_min5_0309.csv", index_col=0)
data = data[['open', 'high', 'low', 'close', 'volume']]
data.index = pd.to_datetime(data.index)

# 각 OHLC 열에 대해 rolling minmax scaling (window=24)
ohlc_features = ['open', 'high', 'low', 'close', 'volume']
for feature in ohlc_features:
    data[feature] = rolling_minmax_scale(data[feature], window=24)

# One-hot 인코딩: 각 OHLC 열은 100구간으로 나눔
data = bin_and_encode(data, ohlc_features, bins=100, drop_original=True)

# 타깃은 원본 close 값 사용 (실험 목적)
data['close_target'] = data['close']
data = data.dropna()

# 최종 입력은 _Bin_ 접미사 열들만 사용
final_input_columns = [col for col in data.columns if '_Bin_' in col]
final_target_column = ['close_target']

data_input = data[final_input_columns]
data_target = data[final_target_column]


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(data_loader), correct / total

def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_and_evaluate(data, num_experiments=16, lookback=24, num_epochs=10):

    final_input_columns = [col for col in data.columns if '_Bin_' in col]
    target_cols = ['close_target']
    
    data_input = data[final_input_columns]
    data_target = data[target_cols]
    
    data_input = data_input.apply(pd.to_numeric).astype(np.float32)
    data_target = data_target.apply(pd.to_numeric).astype(np.float32)
    
    step_size = 31200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    val_acc_list = []
    test_acc_list = []
    
    for exp in range(num_experiments):
        train_start = exp * step_size
        train_end = train_start + step_size * 8
        val_end = train_end + step_size
        test_end = val_end + step_size
        if test_end > len(data_input):
            break
        print(f"Experiment {exp}: 데이터 구간 [{train_start}:{test_end}]")
        print(data)
        
        train_input = data_input.iloc[train_start:train_end]
        train_target = data_target.iloc[train_start:train_end]
        val_input = data_input.iloc[train_end:val_end]
        val_target = data_target.iloc[train_end:val_end]
        test_input = data_input.iloc[val_end:test_end]
        test_target = data_target.iloc[val_end:test_end]
        
        train_dataset = TimeSeriesDataset(train_input, train_target, lookback=lookback)
        val_dataset = TimeSeriesDataset(val_input, val_target, lookback=lookback)
        test_dataset = TimeSeriesDataset(test_input, test_target, lookback=lookback)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        lr = 1e-4
        input_dim = data_input.shape[1]
        model = EncoderOnlyTransformerCustom(input_dim=input_dim, embedding_dim=512, num_layers=6, nhead=8,
                                               ffn_dim=2048, num_classes=2, max_seq_len=lookback).to(device)
        model_path = f"model_experiment_{exp}.pth"
        if exp > 0:
            try:
                model.load_state_dict(torch.load(f"model_experiment_{exp - 1}.pth"))
                print(f"Loaded model from experiment {exp - 1} for fine-tuning.")
            except FileNotFoundError:
                print(f"Model file for experiment {exp - 1} not found. Starting fresh training.")
        
        print(f"Experiment {exp}: Training with lr={lr} (Fine-Tuning)")
        model = train_model(model, train_loader, val_loader, num_epochs, lr, device)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model for experiment {exp}.")
        
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        val_acc_list.append(val_acc)
        print(f"Experiment {exp}: Final Validation Accuracy: {val_acc:.4f}")
        
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        test_acc_list.append(test_acc)
        print(f"Experiment {exp}: Test Accuracy: {test_acc:.4f}")
    
    if val_acc_list:
        avg_val_acc = sum(val_acc_list) / len(val_acc_list)
        avg_test_acc = sum(test_acc_list) / len(test_acc_list)
        print(f"\nFinal Average Validation Accuracy: {avg_val_acc:.4f}")
        print(f"Final Average Test Accuracy: {avg_test_acc:.4f}")
    else:
        print("실험이 한 번도 실행되지 않았습니다.")

train_and_evaluate(data)
