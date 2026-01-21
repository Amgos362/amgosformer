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
# 2. Datetime 인코딩 – Positional (Cyclical) 인코딩
####################################
def encode_datetime_features_positional(data):
    if 'datetime' not in data.columns:
        data['datetime'] = pd.to_datetime(data.index)
    data['hour_sin'] = np.sin(2 * np.pi * data['datetime'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['datetime'].dt.hour / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['datetime'].dt.dayofweek / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['datetime'].dt.dayofweek / 7)
    data['week_of_month'] = (data['datetime'].dt.day - 1) // 7 + 1
    data['week_sin'] = np.sin(2 * np.pi * data['week_of_month'] / 5)
    data['week_cos'] = np.cos(2 * np.pi * data['week_of_month'] / 5)
    data['month_sin'] = np.sin(2 * np.pi * data['datetime'].dt.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['datetime'].dt.month / 12)
    return data

####################################
# 3. 최근 50개 데이터로 rolling minmax scaling (분모 0 방지)
####################################
def rolling_minmax_scale(series, window=32):
    roll_min = series.rolling(window=window, min_periods=window).min()
    roll_max = series.rolling(window=window, min_periods=window).max()
    scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-8)
    return scaled

####################################
# 4. Binning 후 One-Hot 인코딩 (고정 차원)
####################################
def bin_and_encode(data, features, bins=10, drop_original=True):
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
data = pd.read_csv("KRW-ETH_upbit_min60.csv", index_col=0)
data.columns = ['open', 'high', 'low', 'close', 'volume', 'value']
data.index = pd.to_datetime(data.index)
data = calculate_indicators(data)
data = encode_datetime_features_positional(data)

features_to_bin = ['open', 'high', 'low', 'volume', 'value', 'William_R',
                   'ATR', 'OBV', 'Z_Score', 'Entropy', 'SMA_5', 'SMA_10', 
                   'SMA_20', 'SMA_60', 'SMA_120', 'SMA_250', 'RSI', 'BB_Upper', 'BB_Middle', 
                   'BB_Lower', 'MACD', 'Stochastic']
datetime_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos']

# close_target로 원본 close값 보존
data['close_target'] = data['close']

selected_features = features_to_bin + ['close_target'] + datetime_features
data = data[selected_features].dropna()

# rolling scaling 적용 (window=512: 장기 패턴 반영)
for feature in features_to_bin:
    data[feature] = rolling_minmax_scale(data[feature], window=32)
data = data.dropna()

data = bin_and_encode(data, features_to_bin, bins=10, drop_original=True)
data['close_for_binning'] = data['close_target']
data = bin_and_encode(data, ['close_for_binning'], bins=10, drop_original=False)
data.drop(columns=['close_for_binning'], inplace=True)

final_columns = []
for feature in features_to_bin:
    final_columns.extend([f'{feature}_Bin_{i}' for i in range(10)])
final_columns.append('close_target')
final_columns.extend([f'close_for_binning_Bin_{i}' for i in range(10)])
final_columns.extend(datetime_features)
data = data[final_columns]

####################################
# 6. Dataset 정의
####################################
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback=32, target_col='close_target'):
        self.data = data.values
        self.lookback = lookback
        self.target_idx = list(data.columns).index(target_col)

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback, :]
        y = self.data[idx + self.lookback, self.target_idx]
        y_prev = self.data[idx + self.lookback - 1, self.target_idx]
        y_target = 1 if y > y_prev else 0
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

# 7-4. Encoder-Only Transformer 직접 구현
class EncoderOnlyTransformerCustom(nn.Module):
    def __init__(self, input_dim, embedding_dim=512, num_heads=8, num_layers=6, ffn_dim=2048, num_classes=2, max_seq_len=32):
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
# 8. 학습 및 평가 루프 (Fine-tuning 및 Validation Accuracy 출력)
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

def train_and_evaluate(data, num_experiments=16, lookback=32, num_epochs=10):
    input_dim = data.shape[1]
    step_size = 2500  # 이동 단위
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_acc_list = []
    test_acc_list = []

    for exp in range(num_experiments):
        train_start = exp * step_size
        train_end = train_start + step_size * 8   # 훈련 데이터
        val_end = train_end + step_size           # 검증 데이터
        test_end = val_end + step_size            # 테스트 데이터
        if test_end > len(data):
            break

        train_data = data.iloc[train_start:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:test_end]

        train_dataset = TimeSeriesDataset(train_data, lookback=lookback, target_col='close_target')
        val_dataset = TimeSeriesDataset(val_data, lookback=lookback, target_col='close_target')
        test_dataset = TimeSeriesDataset(test_data, lookback=lookback, target_col='close_target')

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Fine-tuning: 초기 실험은 기본 lr, 이후는 미세 조정 lr 사용
        if exp == 0:
            lr = 1e-4
        else:
            lr = 1e-5

        model = EncoderOnlyTransformerCustom(input_dim=input_dim, embedding_dim=512, num_heads=8, 
                                             num_layers=6, ffn_dim=2048, num_classes=2, max_seq_len=32).to(device)

        model_path = f"model_experiment_{exp}.pth"
        if exp > 0:
            try:
                model.load_state_dict(torch.load(f"model_experiment_{exp - 1}.pth"))
                print(f"Loaded model from experiment {exp - 1}.")
            except FileNotFoundError:
                print(f"Model file for experiment {exp - 1} not found. Starting fresh training.")

        print(f"Experiment {exp}: Training with lr={lr}")
        model = train_model(model, train_loader, val_loader, num_epochs, lr, device)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model for experiment {exp}.")

        # 최종 검증 셋 평가
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        val_acc_list.append(val_acc)
        print(f"Experiment {exp}: Final Validation Accuracy: {val_acc:.4f}")

        # 테스트 셋 평가
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        test_acc_list.append(test_acc)
        print(f"Experiment {exp}: Test Accuracy: {test_acc:.4f}")

    # 평균 검증 정확도 및 테스트 정확도 출력
    avg_val_acc = sum(val_acc_list) / len(val_acc_list)
    avg_test_acc = sum(test_acc_list) / len(test_acc_list)

    print(f"\nFinal Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"Final Average Test Accuracy: {avg_test_acc:.4f}")

train_and_evaluate(data)
