import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyupbit
import requests

# --- LINE Notify 설정 ---
TARGET_URL = 'https://notify-api.line.me/api/notify'
TOKEN = "rlMIJRZSatEVj5MLBSqC0iVVRIM7trYKqVbwizh7gUL"
def send_line_notification(message):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"message": message}
    response = requests.post(TARGET_URL, headers=headers, data=data)
    return response

def notify(message):
    print(message)
    send_line_notification(message)

# --- 전처리 함수 ---
def rolling_minmax_scale(series, window=6):
    roll_min = series.rolling(window=window, min_periods=window).min()
    roll_max = series.rolling(window=window, min_periods=window).max()
    scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-8)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    scaled = scaled.fillna(1.0)
    return scaled.clip(upper=1.0)

def bin_and_encode(data, features, bins=100, drop_original=True):
    for feature in features:
        data[f'{feature}_Bin'] = pd.cut(data[feature], bins=bins, labels=False)
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
        y = torch.randn(batch_size, 1, device=device)  # 초기 y: 정규분포 노이즈
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
# OHLC 4개 feature를 100구간으로 binning했으므로 입력 차원은 4*100 = 400
input_dim = 400
lookback = 6
model = DiffusionClassifier(input_dim=input_dim, lookback=lookback, condition_dim=128, num_timesteps=100, hidden_dim=128).to(device)
model_path = "diffusion_model_experiment_14_6.pth"  # 실제 학습된 모델 파일 경로로 수정
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- pyupbit API 설정 ---
access = "x03uKM3uyZ9RNDBHLxauEaeI6X8pdKQq8rHG2wvx"
secret = "e9UyPqJBjCvNDxCmtx7bdPaHtAEgw2Ftxu3UpuJc"
upbit = pyupbit.Upbit(access, secret)
ticker = "KRW-ETH"  # 거래할 코인 티커

# --- 예측 함수 ---
def get_model_prediction():
    """
    pyupbit로 최근 15개의 5분봉 데이터를 가져와 전처리한 후,
    마지막 lookback(6) 봉을 diffusion model의 조건으로 사용하여
    reverse diffusion sampling을 통해 다음 5분봉 상승(1)/하락(0) 예측을 반환.
    """
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=15)
    if df is None or len(df) < lookback:
        notify("예측에 충분한 데이터가 없습니다.")
        return None
    ohlc_features = ['open', 'high', 'low', 'close']
    for feature in ohlc_features:
        df[feature] = rolling_minmax_scale(df[feature], window=6)
    df_processed = bin_and_encode(df.copy(), ohlc_features, bins=100, drop_original=True)
    final_input_columns = [col for col in df_processed.columns if '_Bin_' in col]
    if len(df_processed) < lookback:
        notify("lookback 데이터가 부족합니다.")
        return None
    input_seq = df_processed[final_input_columns].iloc[-lookback:]
    # 모델 입력 형태: [batch, seq_len, features]
    x = torch.tensor(input_seq.values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        y_cont = model.sample(x, device)  # continuous output
        notify("디버그: y_cont = " + str(y_cont.item()))
        prediction = 1 if y_cont.item() >= 0.5 else 0
    return prediction

# --- 글로벌 변수: pending_trade와 거래 통계 ---
pending_trade = None  # 현재 진행 중인 거래 정보 저장 (없으면 None)
stats = {
    'total_trades': 0,
    'win_count': 0,
    'cumulative_return': 1.0,
    'trade_returns': []
}

# --- 거래 실행 함수 (신규 거래 진입) ---
def trade_decision():
    global pending_trade
    prediction = get_model_prediction()
    if prediction is None:
        return
    entry_price = pyupbit.get_current_price(ticker)
    if entry_price is None:
        notify("현재가를 조회하지 못했습니다.")
        return
    if prediction == 1:
        # 상승 예측: KRW 잔고의 50% 사용하여 매수 → 구매 코인 수 = (거래금액 / 진입가)
        krw_balance = upbit.get_balance("KRW")
        if krw_balance is None or krw_balance < 5000:
            notify("매수할 충분한 KRW 잔고가 없습니다.")
            return
        trade_amount_krw = krw_balance * 0.5
        coins_bought = trade_amount_krw / entry_price
        notify(f"[진입] 상승 예측: KRW 잔고의 50%({trade_amount_krw:.0f} KRW)로 매수, 진입가: {entry_price:.2f}, 구매 개수: {coins_bought:.4f} 개")
        upbit.buy_market_order(ticker, trade_amount_krw)  # 실제 주문 실행 (주석 해제)
        pending_trade = {
            'direction': prediction,  # 1: Long
            'entry_price': entry_price,
            'quantity': coins_bought,
            'entry_time': pd.Timestamp.now()
        }
    else:
        # 하락 예측: 보유 코인의 50% 매도
        coin = ticker.split("-")[1]
        coin_balance = upbit.get_balance(coin)
        if coin_balance is None or coin_balance <= 0:
            notify("매도할 보유 코인이 없습니다.")
            return
        trade_amount_coins = coin_balance * 0.5
        notify(f"[진입] 하락 예측: 보유 코인의 50%({trade_amount_coins:.4f} 개) 매도, 진입가: {entry_price:.2f}")
        upbit.sell_market_order(ticker, trade_amount_coins)  # 실제 주문 실행 (주석 해제)
        pending_trade = {
            'direction': prediction,  # 0: Short
            'entry_price': entry_price,
            'quantity': trade_amount_coins,
            'entry_time': pd.Timestamp.now()
        }

# --- 거래 결과 처리 함수 (청산 시점에서 거래 결과 계산) ---
def process_pending_trade():
    global pending_trade, stats
    if pending_trade is None:
        return
    exit_price = pyupbit.get_current_price(ticker)
    if exit_price is None:
        notify("청산 가격을 조회하지 못했습니다.")
        return
    direction = pending_trade['direction']
    entry_price = pending_trade['entry_price']
    fee = 0.0005  # 0.05%
    if direction == 1:  # Long 거래: 상승 예측
        fraction = 0.5
        effective_factor = (exit_price * (1 - fee)) / (entry_price * (1 + fee))
        trade_return = 1 + fraction * (effective_factor - 1)
        trade_type = "Long"
        # hit/miss 판정: Long 거래는 exit_price가 entry_price보다 높아야 hit
        outcome = "Hit" if exit_price > entry_price else "Miss"
    else:  # Short 거래: 하락 예측
        fraction = 0.5
        effective_factor = (entry_price * (1 - fee)) / (exit_price * (1 + fee))
        trade_return = 1 + fraction * (effective_factor - 1)
        trade_type = "Short"
        # hit/miss 판정: Short 거래는 exit_price가 entry_price보다 같거나 낮아야 hit (보합 포함)
        outcome = "Hit" if exit_price <= entry_price else "Miss"
    
    stats['total_trades'] += 1
    if outcome == "Hit":
        stats['win_count'] += 1
    stats['cumulative_return'] *= trade_return
    stats['trade_returns'].append(trade_return)
    quantity = pending_trade.get('quantity', 0)
    notify(f"[청산] {trade_type} 거래 - 진입가: {entry_price:.2f}, 청산가: {exit_price:.2f}, 거래 개수: {quantity:.4f} 개, 거래 수익률: {trade_return:.4f} ({outcome})")
    hit_ratio = stats['win_count'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
    notify(f"거래 횟수: {stats['total_trades']}, Hit Ratio: {hit_ratio:.2%}, 누적 수익률: {stats['cumulative_return']:.4f}\n")
    pending_trade = None

# --- 메인 루프: 새로운 5분봉이 생성될 때마다 업데이트 ---
last_candle_time = None
while True:
    try:
        df = pyupbit.get_ohlcv(ticker, interval="minute5", count=1)
        if df is not None and not df.empty:
            current_candle_time = df.index[-1]
            if last_candle_time is None or current_candle_time > last_candle_time:
                if pending_trade is not None:
                    process_pending_trade()
                notify(f"새로운 5분봉 생성: {current_candle_time}")
                trade_decision()
                last_candle_time = current_candle_time
        else:
            notify("최신 5분봉 데이터를 불러오지 못했습니다.")
    except Exception as e:
        notify(f"에러 발생: {e}")
    time.sleep(10)