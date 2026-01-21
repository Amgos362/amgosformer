import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt

####################################
# 1. 기술적 지표 계산 함수 (변경 없음)
####################################
def calculate_indicators(data):
    data['William_R'] = ta.willr(data['high'], data['low'], data['close'])
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'])
    data['OBV'] = ta.obv(data['close'], data['volume'])
    data['Z_Score'] = (data['close'] - data['close'].rolling(window=20).mean()) / data['close'].rolling(window=20).std()
    data['Entropy'] = ta.entropy(data['close'], length=14)
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_60'] = data['close'].rolling(window=60).mean()
    data['SMA_120'] = data['close'].rolling(window=120).mean()
    data['SMA_250'] = data['close'].rolling(window=250).mean()
    data['RSI'] = ta.rsi(data['close'])
    bb = ta.bbands(data['close'])
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
    macd = ta.macd(data['close'])
    data['MACD'] = macd.iloc[:, 0]
    data['Stochastic'] = ta.stoch(data['high'], data['low'], data['close']).iloc[:, 0]
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
# 2. Datetime Feature One-Hot Encoding (각 feature 128차원)
####################################
def encode_datetime_features_onehot(data, dim=128):
    if 'datetime' not in data.columns:
        data['datetime'] = pd.to_datetime(data.index)
    
    data['hour_of_day'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['week_of_month'] = (data['datetime'].dt.day - 1) // 7 + 1
    data['month'] = data['datetime'].dt.month

    def onehot_with_fixed_dim(series, prefix, dim):
        dummies = pd.get_dummies(series, prefix=prefix)
        expected_cols = [f"{prefix}_{i}" for i in range(dim)]
        dummies = dummies.reindex(columns=expected_cols, fill_value=0)
        return dummies

    hour_one_hot = onehot_with_fixed_dim(data['hour_of_day'], 'Hour', dim)
    day_one_hot = onehot_with_fixed_dim(data['day_of_week'], 'Day', dim)
    week_one_hot = onehot_with_fixed_dim(data['week_of_month'], 'Week', dim)
    month_one_hot = onehot_with_fixed_dim(data['month'], 'Month', dim)
    
    data = pd.concat([data, hour_one_hot, day_one_hot, week_one_hot, month_one_hot], axis=1)
    return data

####################################
# 3. Rolling MinMax Scaling (분모 0 방지)
####################################
def rolling_minmax_scale(series, window=24):
    roll_min = series.rolling(window=window, min_periods=window).min()
    roll_max = series.rolling(window=window, min_periods=window).max()
    scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-8)
    return scaled

####################################
# 4. Binning 후 One-Hot 인코딩 (각 feature를 128차원으로)
####################################
def bin_and_encode(data, features, bins=128, drop_original=True):
    for feature in features:
        data[f'{feature}_Bin'] = pd.cut(data[feature], bins=bins, labels=False)
        one_hot = pd.get_dummies(data[f'{feature}_Bin'], prefix=f'{feature}_Bin')
        expected_columns = [f'{feature}_Bin_{i}' for i in range(bins)]
        one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
        data = pd.concat([data, one_hot], axis=1)
        if drop_original:
            data.drop(columns=[f'{feature}_Bin'], inplace=True)
    data = data.astype(np.float32)
    return data

####################################
# 5. 데이터 로드 및 전처리
####################################
data = pd.read_csv("BTC_upbit_KRW_min5.csv", index_col=0)
data.columns = ['open', 'high', 'low', 'close', 'volume', 'value']
data.index = pd.to_datetime(data.index)

data = calculate_indicators(data)
data = calculate_price_differences(data)   # 추가 feature 계산
data = encode_datetime_features_onehot(data, dim=128)

# 원본 feature와 추가한 가격 차이 feature들을 사용
features_to_bin = ['open', 'high', 'low', 'volume', 'value', 'William_R',
                   'ATR', 'OBV', 'Z_Score', 'Entropy', 'SMA_5', 'SMA_10', 
                   'SMA_20', 'SMA_60', 'SMA_120', 'SMA_250', 'RSI', 'BB_Upper', 'BB_Middle', 
                   'BB_Lower', 'MACD', 'Stochastic',
                   'close_open', 'high_low', 'high_open', 'high_close', 'open_low', 'close_low']

# datetime one-hot 컬럼: 이미 각각 128차원씩 4개 (총 512차원)
datetime_onehot_features = [col for col in data.columns if col.startswith('Hour_') or 
                              col.startswith('Day_') or col.startswith('Week_') or 
                              col.startswith('Month_')]

# 타깃으로 사용할 값들은 모델 타깃으로만 사용하기 위해 보존
data['close_target'] = data['close']
data['open_target'] = data['open']

# 최종 데이터 선택 (여기서 close_target, open_target은 타깃으로만 사용)
data = data[features_to_bin + ['close_target', 'open_target'] + datetime_onehot_features].dropna()

# 각 기술적 및 가격 차이 feature에 대해 rolling scaling 적용 (window=24)
for feature in features_to_bin:
    data[feature] = rolling_minmax_scale(data[feature], window=24)
data = data.dropna()

# 기술적 및 가격 차이 feature들을 128차원의 one-hot 벡터로 변환
data = bin_and_encode(data, features_to_bin, bins=128, drop_original=True)

# 보조로 사용할 close_target에 대해서도 one-hot 인코딩 (128차원)이 필요하면 진행할 수 있으나,
# 여기서는 타깃은 continuous 값으로만 사용합니다.
# 최종 입력 데이터에는 타깃 컬럼은 포함하지 않습니다.

####################################
# 최종 입력/타깃 데이터 구성
####################################
# 입력: 기술적/가격 차이 feature의 one-hot 인코딩 결과와 datetime one-hot 인코딩 결과만
final_input_columns = []
for feature in features_to_bin:
    final_input_columns.extend([f'{feature}_Bin_{i}' for i in range(128)])
final_input_columns.extend(datetime_onehot_features)

# 타깃: open_target과 close_target (continuous)
final_target_columns = ['open_target', 'close_target']

data_input = data[final_input_columns]
data_target = data[final_target_columns]

####################################
# 6-2. Dataset 정의 (입력과 타깃을 별도로 사용)
####################################
class TimeSeriesDataset(Dataset):
    def __init__(self, input_data, target_data, lookback=24):
        self.input_data = input_data.values
        self.target_data = target_data.values  # shape: (N, 2)
        self.lookback = lookback

    def __len__(self):
        return len(self.input_data) - self.lookback

    def __getitem__(self, idx):
        x = self.input_data[idx: idx + self.lookback, :]
        # 타깃은 lookback 이후 시점의 open_target와 close_target 값을 가져와서,
        # candle이 양봉이면 1, 음봉이면 0으로 설정 (양봉: close > open)
        open_val = self.target_data[idx + self.lookback, 0]
        close_val = self.target_data[idx + self.lookback, 1]
        y_target = 1 if close_val > open_val else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_target, dtype=torch.long)

####################################
# 7. Transformer Encoder 직접 구현
####################################
# 7-1. Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim은 num_heads로 나누어떨어져야 합니다."
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out   = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out(out)
        return out

# 7-2. Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 7-3. Transformer Encoder Layer (Self-Attention + FFN + Residual + LayerNorm)
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# 7-4. Encoder-Only Transformer 직접 구현 (lookback=24이므로, max_seq_len=24)
class EncoderOnlyTransformerCustom(nn.Module):
    def __init__(self, input_dim, embedding_dim=512, num_heads=8, num_layers=6, ffn_dim=2048, num_classes=2, max_seq_len=24):
        super(EncoderOnlyTransformerCustom, self).__init__()
        self.token_embedding = nn.Linear(input_dim, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        return self.fc(x)

####################################
# 8. 학습 및 평가 루프 (Fine-Tuning 및 Validation Accuracy 출력)
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
    # 모델 입력은 one-hot 인코딩된 feature들만, 타깃은 continuous open_target과 close_target
    input_cols = []
    for feature in features_to_bin:
        input_cols.extend([f'{feature}_Bin_{i}' for i in range(128)])
    input_cols.extend(datetime_onehot_features)
    target_cols = ['open_target', 'close_target']
    
    # 타깃 컬럼 추가: 원본 open, close 값을 보존 (continuous)
    data['open_target'] = data['open']
    data['close_target'] = data['close']
    
    data_input = data[input_cols]
    data_target = data[target_cols]
    
    step_size = 30000  # 이동 단위
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
        print(data)
        
        train_input = data_input.iloc[train_start:train_end]
        train_target = data_target.iloc[train_start:train_end]
        val_input = data_input.iloc[train_end:val_end]
        val_target = data_target.iloc[train_end:val_end]
        test_input = data_input.iloc[val_end:test_end]
        test_target = data_target.iloc[val_end:test_end]
        
        # Dataset 생성: 입력과 타깃을 따로 전달
        train_dataset = TimeSeriesDataset(train_input, train_target, lookback=lookback)
        val_dataset = TimeSeriesDataset(val_input, val_target, lookback=lookback)
        test_dataset = TimeSeriesDataset(test_input, test_target, lookback=lookback)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Fine-tuning: exp==0일 때는 새로 학습, 이후 구간에서는 이전 모델 파라미터를 그대로 로드하여 미세 조정
        lr = 1e-4
        input_dim = data_input.shape[1]
        model = EncoderOnlyTransformerCustom(input_dim=input_dim, embedding_dim=512, num_heads=8, 
                                               num_layers=6, ffn_dim=2048, num_classes=2, max_seq_len=lookback).to(device)
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
    
    avg_val_acc = sum(val_acc_list) / len(val_acc_list)
    avg_test_acc = sum(test_acc_list) / len(test_acc_list)
    print(f"\nFinal Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"Final Average Test Accuracy: {avg_test_acc:.4f}")

train_and_evaluate(data)
