import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyupbit

# --- 전처리 함수 ---
def rolling_minmax_scale(series, window=3):
    roll_min = series.rolling(window=window, min_periods=window).min()
    roll_max = series.rolling(window=window, min_periods=window).max()
    scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-8)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    scaled = scaled.fillna(1.0)
    return scaled.clip(upper=1.0)

def bin_and_encode(data, features, bins=100, drop_original=True):
    for feature in features:
        data[f'{feature}_Bin'] = pd.cut(data[feature], bins=bins, labels=False, duplicates='drop')
        one_hot = pd.get_dummies(data[f'{feature}_Bin'], prefix=f'{feature}_Bin').astype(np.int32)
        expected_columns = [f'{feature}_Bin_{i}' for i in range(bins)]
        one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
        data = pd.concat([data, one_hot], axis=1)
        if drop_original:
            data.drop(columns=[f'{feature}_Bin'], inplace=True)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        data[col] = data[col].astype(np.float32)
    return data


# --- Diffusion Model 구성 ---
class ConditionEncoder(nn.Module):
    def __init__(self, input_dim, lookback, condition_dim):
        super(ConditionEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * lookback, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
    def forward(self, x):
        # x: [batch, lookback, input_dim]
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)
        return self.fc(x)

class DiffusionClassifier(nn.Module):
    def __init__(self, input_dim, lookback, condition_dim=128, num_timesteps=100, hidden_dim=128):
        super(DiffusionClassifier, self).__init__()
        self.num_timesteps = num_timesteps
        # diffusion 스케줄 (선형 beta schedule)
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # 조건 인코더: 시계열 window를 임베딩
        self.condition_encoder = ConditionEncoder(input_dim, lookback, condition_dim)
        # timestep 임베딩
        self.time_embedding = nn.Embedding(num_timesteps, hidden_dim)
        # 노이즈 예측 네트워크
        self.model = nn.Sequential(
            nn.Linear(1 + condition_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_condition, y_noisy, t):
        # x_condition: [batch, lookback, input_dim]
        # y_noisy: [batch, 1]
        # t: [batch]
        cond = self.condition_encoder(x_condition)           # [batch, condition_dim]
        t_emb = self.time_embedding(t)                         # [batch, hidden_dim]
        inp = torch.cat([y_noisy, cond, t_emb], dim=1)          # [batch, 1 + condition_dim + hidden_dim]
        predicted_noise = self.model(inp)                      # [batch, 1]
        return predicted_noise
    
    def sample(self, x_condition, device):
        """
        reverse diffusion 과정을 통해 조건 x_condition에 대해 예측값을 샘플링.
        최종 출력은 continuous 값으로, 임계값 0.5를 기준으로 상승(1) 또는 하락(0) 결정.
        """
        batch_size = x_condition.size(0)
        y = torch.randn(batch_size, 1, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self.forward(x_condition, y, t_tensor)
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            y = (1 / torch.sqrt(alpha)) * (y - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            if t > 0:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta) * noise
        return y

# --- 모델 로드 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# OHLC 4개 feature를 100구간으로 binning하므로 입력 차원은 4*100 = 400
input_dim = 400
lookback = 3  # 백테스트 시 사용할 lookback 기간 (5분봉 기준 3개)
model = DiffusionClassifier(input_dim=input_dim, lookback=lookback, condition_dim=128, num_timesteps=100, hidden_dim=128).to(device)
model_path = "diffusion_model_experiment_14.pth"  # 실제 학습된 모델 파일 경로로 수정
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 백테스팅용 예측 함수 ---
def get_prediction_from_window(df, idx, lookback, device):
    """
    df: OHLC 데이터프레임 (open, high, low, close 컬럼 포함)
    idx: 거래를 진행할 시점 (해당 인덱스의 캔들로 거래한다고 가정)
    lookback: 입력으로 사용할 캔들 개수
    """
    ohlc_features = ['open', 'high', 'low', 'close']
    # 거래 전 lookback 캔들 선택
    window_df = df.iloc[idx - lookback: idx].copy()
    # 각 feature에 대해 rolling scaling 적용
    for feature in ohlc_features:
        window_df[feature] = rolling_minmax_scale(window_df[feature], window=lookback)
    window_df = bin_and_encode(window_df, ohlc_features, bins=100, drop_original=True)
    final_input_columns = [col for col in window_df.columns if '_Bin_' in col]
    if len(final_input_columns) == 0:
        return None
    input_seq = window_df[final_input_columns].iloc[-lookback:]
    x = torch.tensor(input_seq.values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        y_cont = model.sample(x, device)
        prediction = 1 if y_cont.item() >= 0.5 else 0
    return prediction

# --- 백테스팅 함수 ---
def backtest_trading(df, model, lookback, device):
    """
    df: 과거 OHLC 데이터 (open, high, low, close 컬럼)
    모델을 통해 각 거래 시점의 예측에 따라 롱/숏 거래를 시뮬레이션.
    롱: entry = open, exit = close, 수익률 = close/open
    숏: entry = open, exit = close, 수익률 = open/close
    각 거래별 승리 여부(hit ratio)는 수익률 > 1 인 경우로 판단.
    누적 수익률은 모든 거래 수익률의 곱으로 계산.
    """
    total_trades = 0
    win_count = 0
    cumulative_return = 1.0
    trade_returns = []
    predictions = []
    
    # 인덱스 lookback부터 시작 (각 거래 시점에 대해 이전 lookback 캔들 사용)
    for i in range(lookback, len(df)):
        prediction = get_prediction_from_window(df, i, lookback, device)
        if prediction is None:
            continue
        predictions.append(prediction)
        trade_candle = df.iloc[i]
        if prediction == 1:
            # 예측이 상승이면 롱 거래: entry = open, exit = close
            trade_ret = trade_candle['close'] / trade_candle['open']
        else:
            # 예측이 하락이면 숏 거래: entry = open, exit = close
            trade_ret = trade_candle['open'] / trade_candle['close']
        trade_returns.append(trade_ret)
        total_trades += 1
        cumulative_return *= trade_ret
        # 거래 승리 판정: 수익률이 1보다 큰 경우 승리
        if trade_ret > 1:
            win_count += 1

    hit_ratio = win_count / total_trades if total_trades > 0 else 0
    return hit_ratio, cumulative_return, total_trades, trade_returns, predictions

# --- 백테스팅 실행 ---
# 예를 들어, CSV 파일에 저장된 과거 데이터 사용 (OHLC 데이터)
data_path = "KRW-ETH_upbit_min60.csv"  # 파일 경로에 맞게 수정
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)
df = df[['open', 'high', 'low', 'close']]

# 백테스트 실행 (예: 전체 데이터에 대해)
hit_ratio, cum_return, num_trades, trade_returns, preds = backtest_trading(df, model, lookback, device)
print("총 거래 횟수:", num_trades)
print("Hit Ratio (승률): {:.2%}".format(hit_ratio))
print("누적 수익률:", cum_return)
