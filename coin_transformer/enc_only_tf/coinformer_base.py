import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import math


####################################
# 1. 기술적 지표 계산 함수 (수정됨)
####################################
def calculate_indicators(data):
    # SMA: 5,10,20,60,120,250
    data['SMA_5']   = data['close'].rolling(window=5).mean()
    data['SMA_10']  = data['close'].rolling(window=10).mean()
    data['SMA_20']  = data['close'].rolling(window=20).mean()
    data['SMA_60']  = data['close'].rolling(window=60).mean()
    data['SMA_120'] = data['close'].rolling(window=120).mean()
    data['SMA_250'] = data['close'].rolling(window=250).mean()
    
    # RSI (14일)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 볼린저밴드 (20일 MA, 2σ)
    ma20 = data['close'].rolling(window=20).mean()
    std20 = data['close'].rolling(window=20).std()
    data['BB_Middle'] = ma20
    data['BB_Upper'] = ma20 + 2 * std20
    data['BB_Lower'] = ma20 - 2 * std20
    # 볼린저밴드 폭
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    
    # 스토캐스틱 slow: 14일, 3일 SMA 이용 → 결과 0~100
    lowest14 = data['low'].rolling(window=14).min()
    highest14 = data['high'].rolling(window=14).max()
    fast_K = (data['close'] - lowest14) / (highest14 - lowest14) * 100
    slow_K = fast_K.rolling(window=3).mean()
    slow_D = slow_K.rolling(window=3).mean()
    data['Stochastic_K'] = slow_K
    data['Stochastic_D'] = slow_D
    
    # 모멘텀: 당일 종가 - 10일 전 종가
    data['Momentum'] = data['close'] - data['close'].shift(10)
    # 모멘텀 가속도: (당일 종가 - 5일 전 종가) / (5일 전 종가 - 10일 전 종가)
    data['Momentum_acceleration'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) - data['close'].shift(10))
    
    return data.dropna()

####################################
# 1-2. 추가 feature 계산 (가격 차이)
####################################
def calculate_price_differences(data):
    data['close_open'] = data['close'] - data['open']
    data['high_low'] = data['high'] - data['low']
    data['high_open'] = data['high'] - data['open']
    data['high_close'] = data['high'] - data['close']
    data['open_low'] = data['open'] - data['low']
    data['close_low'] = data['close'] - data['low']
    return data

####################################
# 2. Datetime Feature One-Hot Encoding (각 feature 100차원)
####################################
def encode_datetime_features_onehot(data):
    if 'datetime' not in data.columns:
        data['datetime'] = pd.to_datetime(data.index)
    
    # 원본 datetime 정보 생성
    data['hour_of_day'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['week_of_month'] = ((data['datetime'].dt.day - 1) // 7) + 1
    data['month'] = data['datetime'].dt.month

    # 각 컬럼에 대해 one-hot 인코딩 (0,1 정수형)
    hour_dummies = pd.get_dummies(data['hour_of_day'], prefix='Hour').astype(int)
    day_dummies = pd.get_dummies(data['day_of_week'], prefix='Day').astype(int)
    week_dummies = pd.get_dummies(data['week_of_month'], prefix='Week').astype(int)
    month_dummies = pd.get_dummies(data['month'], prefix='Month').astype(int)
    
    # 시간은 0~23, 요일은 0~6, 주는 1~5, 월은 1~12로 고정된 컬럼을 보장
    for i in range(24):
        col_name = f'Hour_{i}'
        if col_name not in hour_dummies.columns:
            hour_dummies[col_name] = 0
    hour_dummies = hour_dummies[[f'Hour_{i}' for i in range(24)]]
    
    for i in range(7):
        col_name = f'Day_{i}'
        if col_name not in day_dummies.columns:
            day_dummies[col_name] = 0
    day_dummies = day_dummies[[f'Day_{i}' for i in range(7)]]
    
    for i in range(1, 6):
        col_name = f'Week_{i}'
        if col_name not in week_dummies.columns:
            week_dummies[col_name] = 0
    week_dummies = week_dummies[[f'Week_{i}' for i in range(1, 6)]]
    
    for i in range(1, 13):
        col_name = f'Month_{i}'
        if col_name not in month_dummies.columns:
            month_dummies[col_name] = 0
    month_dummies = month_dummies[[f'Month_{i}' for i in range(1, 13)]]
    
    # 원본 데이터와 one-hot 인코딩된 컬럼들을 결합
    dt_dummies = pd.concat([hour_dummies, day_dummies, week_dummies, month_dummies], axis=1)
    data = pd.concat([data, dt_dummies], axis=1)
    return data

####################################
# 추가: Datetime Positional Encoding (sin, cos)
####################################
def add_datetime_positional_encoding(data):
    if data['hour_of_day'].notna().any():
        data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
    else:
        data['hour_sin'] = 0
        data['hour_cos'] = 0
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['week_sin'] = np.sin(2 * np.pi * data['week_of_month'] / 5)
    data['week_cos'] = np.cos(2 * np.pi * data['week_of_month'] / 5)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    return data



####################################
# 3. Rolling MinMax Scaling (분모 0 방지)
####################################
def rolling_minmax_scale(series, window=24):
    roll_min = series.rolling(window=window, min_periods=window).min()
    roll_max = series.rolling(window=window, min_periods=window).max()
    scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-8)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    scaled = scaled.fillna(1.0)
    scaled = scaled.clip(upper=1.0)
    return scaled

####################################
# 4. Binning 후 One-Hot 인코딩 (각 feature를 100차원으로)
####################################
def bin_and_encode(data, features, bins=100, drop_original=True):
    for feature in features:
        data[f'{feature}_Bin'] = pd.cut(data[feature], bins=bins, labels=False)
        one_hot = pd.get_dummies(data[f'{feature}_Bin'], prefix=f'{feature}_Bin').astype(np.int32)
        expected_columns = [f'{feature}_Bin_{i}' for i in range(bins)]
        one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
        data = pd.concat([data, one_hot], axis=1)
        if drop_original:
            data.drop(columns=[f'{feature}_Bin'], inplace=True)
    # 개별 열 단위로 숫자형 열을 float32로 변환
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        data[col] = data[col].astype(np.float32)
    return data



####################################
# 추가: Order(순위) 인코딩 함수
####################################
def add_combined_order_encoding(data, group_cols):
    # 각 행별 순위를 계산 (낮은 값 → 낮은 순위, 1부터 시작)
    rank_df = data[group_cols].rank(method='min', axis=1).astype(int)
    onehot_list = []
    # 각 feature에 대해 1~10의 순위를 10차원 one-hot 벡터로 인코딩 (정수형으로 강제)
    for col in group_cols:
        dummies = pd.get_dummies(rank_df[col], prefix=f"{col}_order").astype(np.int32)
        expected_cols = [f"{col}_order_{i}" for i in range(1, 11)]
        dummies = dummies.reindex(columns=expected_cols, fill_value=0).astype(np.int32)
        onehot_list.append(dummies)
    combined = pd.concat(onehot_list, axis=1)
    data = pd.concat([data, combined], axis=1)
    return data


def onehot_order_encoding(data, order_cols, num_categories):
    for col in order_cols:
        dummies = pd.get_dummies(data[col], prefix=col).astype(np.int32)
        expected_cols = [f"{col}_{i}" for i in range(1, num_categories+1)]
        dummies = dummies.reindex(columns=expected_cols, fill_value=0)
        data = pd.concat([data, dummies], axis=1)
        data.drop(columns=[col], inplace=True)
    return data


####################################
# 5. 데이터 로드 및 전처리
####################################
data = pd.read_csv("BTC_upbit_KRW_min60.csv", index_col=0)
data.columns = ['open', 'high', 'low', 'close', 'volume', 'value']
data.index = pd.to_datetime(data.index)
data['volume'] = np.log(data['volume'] + 1)
data['value'] = np.log(data['value'] + 1)

# 기술적 지표 및 추가 feature 계산
data = calculate_indicators(data)
data = calculate_price_differences(data)
data = encode_datetime_features_onehot(data)
data = add_datetime_positional_encoding(data)

data['close_target'] = data['close']  # 원본 close 열을 보존
####################################
# [인코딩 대상 feature 목록 업데이트]
####################################
# 사용할 feature:
# OHLC: open, high, low, close  
# 거래량/거래대금: volume, value  
# 지표: RSI, BB_Upper, BB_Middle, BB_Lower, BB_Width, Stochastic_K, Stochastic_D, Momentum, Momentum_acceleration  
# 가격 차이: close_open, high_low, high_open, high_close, open_low, close_low  
# SMA: SMA_5, SMA_10, SMA_20, SMA_60, SMA_120, SMA_250
features_to_bin = ['open', 'high', 'low', 'close', 'volume', 'value', 'RSI', 
                   'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'Stochastic_K', 'Stochastic_D',
                   'Momentum', 'Momentum_acceleration',
                   'close_open', 'high_low', 'high_open', 'high_close', 'open_low', 'close_low',
                   'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60', 'SMA_120', 'SMA_250']

# 그룹 A: 상승률 기반 인코딩 (pct_change) → rolling minmax scaling → 100구간 binning
for feature in features_to_bin:
    col_pct = feature + '_pct'
    data[col_pct] = data[feature].pct_change() * 100
    data[col_pct] = rolling_minmax_scale(data[col_pct], window=24)
data = data.dropna()
features_pct = [f + '_pct' for f in features_to_bin]
data = bin_and_encode(data, features_pct, bins=100, drop_original=True)

# 그룹 B: 원래 값 기반 인코딩 → rolling minmax scaling → 100구간 binning
for feature in features_to_bin:
    col_minmax = feature + '_minmax'
    data[col_minmax] = rolling_minmax_scale(data[feature], window=24)
features_minmax = [f + '_minmax' for f in features_to_bin]
data = bin_and_encode(data, features_minmax, bins=100, drop_original=True)

# 그룹 C: 고정 범위 인코딩 (OHLC와 SMA 대상)
ohlc_fixed = ['open', 'high', 'low', 'close']
sma_fixed = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_60', 'SMA_120', 'SMA_250']
for feat in ohlc_fixed + sma_fixed:
    col_fixed = feat + '_fixed'
    data[col_fixed] = data[feat].pct_change() * 100
    data[col_fixed] = data[col_fixed].clip(lower=-30, upper=30)
    data[f'{col_fixed}_Bin'] = pd.cut(data[col_fixed], bins=np.linspace(-30,30,101), labels=False)
    one_hot = pd.get_dummies(data[f'{col_fixed}_Bin'], prefix=f'{col_fixed}_Bin')
    expected_columns = [f'{col_fixed}_Bin_{i}' for i in range(100)]
    one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
    data = pd.concat([data, one_hot], axis=1)
    data.drop(columns=[f'{col_fixed}_Bin'], inplace=True)

# 그룹 D: 거래량 및 거래대금 – 이미 log 변환된 값에 대해 처리
vol_fields = ['volume', 'value']
for feat in vol_fields:
    data[feat + '_log'] = rolling_minmax_scale(data[feat], window=24)
data = bin_and_encode(data, [f + '_log' for f in vol_fields], bins=100, drop_original=True)

# 그룹 E: RSI, Stochastic 계열 – 값이 이미 0~100 범위이므로 그대로 인코딩
rsi_stoch = ['RSI', 'Stochastic_K', 'Stochastic_D']
data = bin_and_encode(data, rsi_stoch, bins=100, drop_original=True)

# 그룹 F: OHLC 가격 차이 (6개) – pct_change 기반 처리
price_diff = ['close_open', 'high_low', 'high_open', 'high_close', 'open_low', 'close_low']
for feat in price_diff:
    data[feat + '_pct'] = data[feat].pct_change() * 100
    data[feat + '_pct'] = rolling_minmax_scale(data[feat + '_pct'], window=24)
data = bin_and_encode(data, [f + '_pct' for f in price_diff], bins=100, drop_original=True)

# 그룹 G: Order(순위) 인코딩
# OHLC 순위 (open, high, low, close) → 4 범주
combined_order_features = ['open', 'high', 'low', 'close', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60', 'SMA_120', 'SMA_250']
data = add_combined_order_encoding(data, combined_order_features)

####################################
# 최종 입력 데이터 구성
####################################
# 최종 입력은 _Bin_ 컬럼, datetime one-hot, positional encoding, 그리고 order 인코딩 컬럼들
final_input_columns = [col for col in data.columns if '_Bin_' in col]
datetime_onehot_features = [col for col in data.columns if col.startswith('Hour_') or 
                              col.startswith('Day_') or col.startswith('Week_') or 
                              col.startswith('Month_')]
datetime_positional_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos']
order_encoding_features = [col for col in data.columns if '_order_' in col]
final_input_columns.extend(datetime_onehot_features)
final_input_columns.extend(datetime_positional_features)
final_input_columns.extend(order_encoding_features)
final_target_column = ['close_target']

data_input = data[final_input_columns]
data_target = data[final_target_column]

####################################
# 6-2. Dataset 정의
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

####################################
# 7. Transformer Encoder (torch.nn 이용)
####################################
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
        x = x[-1, :, :]
        return self.fc(x)

####################################
# 8. 학습 및 평가 루프
####################################
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
    # 최종 입력: 모든 _Bin_ 컬럼 + datetime one-hot + positional encoding + order 인코딩 컬럼
    final_input_columns = [col for col in data.columns if '_Bin_' in col]
    datetime_onehot_features = [col for col in data.columns if col.startswith('Hour_') or 
                                  col.startswith('Day_') or col.startswith('Week_') or 
                                  col.startswith('Month_')]
    datetime_positional_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos']
    order_encoding_features = [col for col in data.columns if '_order_' in col]
    final_input_columns.extend(datetime_onehot_features)
    final_input_columns.extend(datetime_positional_features)
    final_input_columns.extend(order_encoding_features)
    target_cols = ['close_target']
    
    data_input = data[final_input_columns]
    data_target = data[target_cols]
    
    data_input = data_input.apply(pd.to_numeric).astype(np.float32)
    data_target = data_target.apply(pd.to_numeric).astype(np.float32)
    
    step_size = 2500
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
        data.to_csv('AAPL_one_hot.csv')
        
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

# 참조용 리스트 (최종 구성에는 _Bin_ 컬럼들이 포함됨)
features_pct = [f + '_pct' for f in features_to_bin]
features_minmax = [f + '_minmax' for f in features_to_bin]

train_and_evaluate(data)
