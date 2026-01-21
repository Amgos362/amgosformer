import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def calculate_indicators(data):
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_60'] = data['close'].rolling(window=60).mean()
    data['SMA_120'] = data['close'].rolling(window=120).mean()
    return data.dropna()

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

##############################################
# Diffusion 모델을 위한 Dataset 정의
##############################################
class DiffusionTimeSeriesDataset(Dataset):
    def __init__(self, input_data, target_data, lookback=24):
        self.input_data = input_data.values
        self.target_data = target_data.values
        self.lookback = lookback

    def __len__(self):
        return len(self.input_data) - self.lookback

    def __getitem__(self, idx):
        # 조건 데이터: lookback window
        x = self.input_data[idx: idx + self.lookback, :]
        # 타깃: close_target 값을 이용해 상승이면 1, 하락이면 0으로 설정 (float형)
        y = self.target_data[idx + self.lookback, 0]
        y_prev = self.target_data[idx + self.lookback - 1, 0]
        label = 1.0 if y > y_prev else 0.0
        return torch.tensor(x, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

##############################################
# Condition Encoder: 시계열 window를 벡터로 인코딩
##############################################
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
        x = x.view(batch_size, -1)
        return self.fc(x)

##############################################
# DiffusionClassifier: diffusion process를 통한 노이즈 예측 모델
##############################################
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
        
        # 조건 인코더: 시계열 데이터를 조건으로 임베딩
        self.condition_encoder = ConditionEncoder(input_dim, lookback, condition_dim)
        # timestep 임베딩
        self.time_embedding = nn.Embedding(num_timesteps, hidden_dim)
        # 노이즈 예측 네트워크: 입력은 [y_noisy, condition, timestep embedding]
        self.model = nn.Sequential(
            nn.Linear(1 + condition_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_condition, y_noisy, t):
        # x_condition: [batch, lookback, input_dim]
        # y_noisy: [batch, 1] - 노이즈가 추가된 타깃
        # t: [batch] - timestep 인덱스
        cond = self.condition_encoder(x_condition)           # [batch, condition_dim]
        t_emb = self.time_embedding(t)                         # [batch, hidden_dim]
        inp = torch.cat([y_noisy, cond, t_emb], dim=1)          # [batch, 1+condition_dim+hidden_dim]
        predicted_noise = self.model(inp)                      # [batch, 1]
        return predicted_noise
    
    def sample(self, x_condition, device):
        """
        reverse diffusion 과정을 통해 조건 x_condition에 대해 예측값을 샘플링
        최종 출력은 continuous 값으로, 임계값 0.5로 분류 가능함.
        """
        batch_size = x_condition.size(0)
        # 초기 y: 정규분포 노이즈
        y = torch.randn(batch_size, 1, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self.forward(x_condition, y, t_tensor)
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            # DDPM 업데이트: 간단화된 형태
            y = (1 / torch.sqrt(alpha)) * (y - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            # t > 0이면 약간의 노이즈 추가
            if t > 0:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta) * noise
        return y

##############################################
# Diffusion 모델 학습 및 평가 함수
##############################################
def train_diffusion_model(model, dataloader, num_epochs, device, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in dataloader:
            x = x.to(device)  # [batch, lookback, input_dim]
            y = y.to(device)  # [batch, 1] (0.0 or 1.0)
            batch_size = x.size(0)
            # 각 배치마다 timestep t를 균등 샘플링
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
            # 해당 timestep에 따른 누적 알파값
            alphas_cumprod_t = model.alphas_cumprod[t].view(batch_size, 1)
            # 노이즈 샘플링
            noise = torch.randn_like(y)
            # y_noisy = sqrt(alpha_cumprod)*y + sqrt(1-alpha_cumprod)*noise
            y_noisy = torch.sqrt(alphas_cumprod_t) * y + torch.sqrt(1 - alphas_cumprod_t) * noise
            optimizer.zero_grad()
            predicted_noise = model(x, y_noisy, t)
            loss = mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def evaluate_diffusion_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            # reverse diffusion 과정을 통해 예측값 샘플링
            y_sampled = model.sample(x, device)
            # 0.5 기준으로 분류
            y_pred = (y_sampled >= 0.5).float()
            correct += (y_pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Evaluation Accuracy: {acc:.4f}")
    return acc

##############################################
# 데이터 로드 및 전처리 (OHLC 4개 데이터 사용)
##############################################
data = pd.read_csv("KRW-ETH_upbit_min60.csv", index_col=0)
data = calculate_indicators(data)
data = data[['open', 'high', 'low', 'close']]
data.index = pd.to_datetime(data.index)

ohlc_features = ['open', 'high', 'low', 'close']
for feature in ohlc_features:
    data[feature] = rolling_minmax_scale(data[feature], window=24)

data = bin_and_encode(data, ohlc_features, bins=100, drop_original=True)
# 타깃은 원본 close 값 사용 (실험 목적)
data['close_target'] = data['close']
data = data.dropna()

# 최종 입력: _Bin_ 접미사가 있는 열들만 사용
final_input_columns = [col for col in data.columns if '_Bin_' in col]
final_target_column = ['close_target']

data_input = data[final_input_columns]
data_target = data[final_target_column]

##############################################
# 실험 실행: Diffusion Model 기반 주가 상승/하락 예측
##############################################
def train_and_evaluate_diffusion(data, num_experiments=15, lookback=24, num_epochs=10):
    final_input_columns = [col for col in data.columns if '_Bin_' in col]
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
        print(f"\nExperiment {exp}: 데이터 구간 [{train_start}:{test_end}]")
        
        train_input = data_input.iloc[train_start:train_end]
        train_target = data_target.iloc[train_start:train_end]
        val_input = data_input.iloc[train_end:val_end]
        val_target = data_target.iloc[train_end:val_end]
        test_input = data_input.iloc[val_end:test_end]
        test_target = data_target.iloc[val_end:test_end]
        
        train_dataset = DiffusionTimeSeriesDataset(train_input, train_target, lookback=lookback)
        val_dataset = DiffusionTimeSeriesDataset(val_input, val_target, lookback=lookback)
        test_dataset = DiffusionTimeSeriesDataset(test_input, test_target, lookback=lookback)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        input_dim = train_input.shape[1]
        model = DiffusionClassifier(input_dim=input_dim, lookback=lookback, 
                                    condition_dim=128, num_timesteps=100, hidden_dim=128).to(device)
        model_path = f"diffusion_model_experiment_{exp}.pth"
        if exp > 0:
            try:
                model.load_state_dict(torch.load(f"diffusion_model_experiment_{exp - 1}.pth"))
                print(f"Loaded model from experiment {exp - 1} for fine-tuning.")
            except FileNotFoundError:
                print(f"Model file for experiment {exp - 1} not found. Starting fresh training.")
        
        print(f"Experiment {exp}: Training Diffusion Model")
        train_diffusion_model(model, train_loader, num_epochs, device, lr=1e-4)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model for experiment {exp}.")
        
        print("Validation Evaluation:")
        val_acc = evaluate_diffusion_model(model, val_loader, device)
        val_acc_list.append(val_acc)
        
        print("Test Evaluation:")
        test_acc = evaluate_diffusion_model(model, test_loader, device)
        test_acc_list.append(test_acc)
    
    if val_acc_list:
        avg_val_acc = sum(val_acc_list) / len(val_acc_list)
        avg_test_acc = sum(test_acc_list) / len(test_acc_list)
        print(f"\nFinal Average Validation Accuracy: {avg_val_acc:.4f}")
        print(f"Final Average Test Accuracy: {avg_test_acc:.4f}")
    else:
        print("실험이 한 번도 실행되지 않았습니다.")

# 최종적으로 Diffusion Model 실험 실행
train_and_evaluate_diffusion(data, num_experiments=15, lookback=24, num_epochs=10)
