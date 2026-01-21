import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt

####################################
# 전처리 함수 정의
####################################
# rolling minmax scaling 함수 (window=24)
# 주의: 입력 feature는 스케일링되어 [0,1] 범위 내 값으로 변환됨
def rolling_minmax_scale(series, window=24):
    roll_min = series.rolling(window=window, min_periods=window).min()
    roll_max = series.rolling(window=window, min_periods=window).max()
    scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-8)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    scaled = scaled.fillna(1.0)
    return scaled.clip(upper=1.0)

# binning 및 one-hot 인코딩 함수 (OHLC 열, bins=100)
def bin_and_encode(data, features, bins=100, drop_original=True):
    for feature in features:
        # pd.cut으로 최소~최대 구간을 나눔
        data[f'{feature}_Bin'] = pd.cut(data[feature], bins=bins, labels=False)
        one_hot = pd.get_dummies(data[f'{feature}_Bin'], prefix=f'{feature}_Bin').astype(np.int32)
        expected_columns = [f'{feature}_Bin_{i}' for i in range(bins)]
        one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
        data = pd.concat([data, one_hot], axis=1)
        if drop_original:
            data.drop(columns=[f'{feature}_Bin'], inplace=True)
    # 모든 숫자형 열을 float32로 변환
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        data[col] = data[col].astype(np.float32)
    return data

####################################
# Regression Dataset 정의
####################################
class TimeSeriesDatasetRegression(Dataset):
    def __init__(self, input_data, target_data, lookback=24):
        # input_data: one-hot 인코딩된 feature, target_data: 스케일된 실제 close 값
        self.input_data = input_data.values
        self.target_data = target_data.values  # 연속형 값
        self.lookback = lookback

    def __len__(self):
        return len(self.input_data) - self.lookback

    def __getitem__(self, idx):
        # 입력 시퀀스: lookback 길이의 행들
        x = self.input_data[idx: idx + self.lookback, :]
        # 타깃: 바로 다음 시점의 close_target (연속형 값)
        y = self.target_data[idx + self.lookback, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

####################################
# Regression 모델 정의
####################################
class EncoderOnlyTransformerRegression(nn.Module):
    def __init__(self, input_dim, embedding_dim=512, num_layers=6, nhead=8, 
                 ffn_dim=2048, max_seq_len=24):
        super(EncoderOnlyTransformerRegression, self).__init__()
        self.token_embedding = nn.Linear(input_dim, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=ffn_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)  # 회귀를 위한 출력 (실제 close 예측)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        x = x.transpose(0, 1)  # Transformer 입력: [seq_len, batch, features]
        x = self.transformer_encoder(x)
        return self.fc(x[-1, :, :])  # 마지막 타임스탭의 출력

####################################
# 데이터 로드 및 전처리 (OHLC 데이터)
####################################
# CSV 파일 로드 – OHLC 4개 데이터만 사용
data = pd.read_csv("BTC_upbit_KRW_min60.csv", index_col=0)
data = data[['open', 'high', 'low', 'close']]
data.index = pd.to_datetime(data.index)

# 각 OHLC 열에 대해 rolling minmax scaling (window=24)
ohlc_features = ['open', 'high', 'low', 'close']
for feature in ohlc_features:
    data[feature] = rolling_minmax_scale(data[feature], window=24)

# One-hot 인코딩: OHLC 열은 100구간으로 나눔 (입력 feature로 사용)
data_encoded = bin_and_encode(data.copy(), ohlc_features, bins=100, drop_original=True)

# 타깃은 원본 close 값을 사용 (실험 목적: 여기서는 스케일된 close 값)
data['close_target'] = data['close']
data = data.dropna()

# 최종 입력은 one-hot 인코딩된 열들만 사용
final_input_columns = [col for col in data_encoded.columns if '_Bin_' in col]
final_target_column = ['close_target']

data_input = data_encoded[final_input_columns]
data_target = data[final_target_column]

####################################
# 평가 및 학습 함수 (Regression)
####################################
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    n_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)  # 출력 shape: (batch, 1)
            loss = criterion(outputs, y.unsqueeze(1))
            total_loss += loss.item() * y.size(0)
            n_samples += y.size(0)
    mse = total_loss / n_samples
    rmse = sqrt(mse)
    return mse, rmse

def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            n_samples += y.size(0)
        train_loss = total_loss / n_samples
        
        val_mse, val_rmse = evaluate_model(model, val_loader, device)
        scheduler.step(val_mse)
        print(f"Epoch {epoch+1}/{num_epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_mse:.6f}, Val RMSE: {val_rmse:.6f}")
        
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_and_evaluate(data_input, data_target, num_experiments=16, lookback=24, num_epochs=10):
    # 입력 데이터: one-hot 인코딩된 값, 타깃: close_target (연속형 값)
    data_input = data_input.apply(pd.to_numeric).astype(np.float32)
    data_target = data_target.apply(pd.to_numeric).astype(np.float32)
    
    step_size = 2500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    val_rmse_list = []
    test_rmse_list = []
    
    for exp in range(num_experiments):
        train_start = exp * step_size
        train_end = train_start + step_size * 8
        val_end = train_end + step_size
        test_end = val_end + step_size
        if test_end > len(data_input):
            break
        print(f"\nExperiment {exp}: 데이터 구간 [{train_start}:{test_end}]")
        
        train_input = data_input.iloc[train_start:train_end]
        train_target = data_target.iloc[train_start:train_end]
        val_input = data_input.iloc[train_end:val_end]
        val_target = data_target.iloc[train_end:val_end]
        test_input = data_input.iloc[val_end:test_end]
        test_target = data_target.iloc[val_end:test_end]
        
        train_dataset = TimeSeriesDatasetRegression(train_input, train_target, lookback=lookback)
        val_dataset = TimeSeriesDatasetRegression(val_input, val_target, lookback=lookback)
        test_dataset = TimeSeriesDatasetRegression(test_input, test_target, lookback=lookback)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        lr = 1e-4
        input_dim = data_input.shape[1]
        model = EncoderOnlyTransformerRegression(
            input_dim=input_dim, embedding_dim=512, num_layers=6, nhead=8,
            ffn_dim=2048, max_seq_len=lookback).to(device)
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
        
        _, val_rmse = evaluate_model(model, val_loader, device)
        val_rmse_list.append(val_rmse)
        print(f"Experiment {exp}: Final Validation RMSE: {val_rmse:.6f}")
        
        _, test_rmse = evaluate_model(model, test_loader, device)
        test_rmse_list.append(test_rmse)
        print(f"Experiment {exp}: Test RMSE: {test_rmse:.6f}")
    
    if val_rmse_list:
        avg_val_rmse = sum(val_rmse_list) / len(val_rmse_list)
        avg_test_rmse = sum(test_rmse_list) / len(test_rmse_list)
        print(f"\nFinal Average Validation RMSE: {avg_val_rmse:.6f}")
        print(f"Final Average Test RMSE: {avg_test_rmse:.6f}")
    else:
        print("실험이 한 번도 실행되지 않았습니다.")

# 학습 실행
train_and_evaluate(data_input, data_target)
