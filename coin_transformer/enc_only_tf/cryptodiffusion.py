import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt

##############################################
# 데이터 전처리: rolling minmax scaling 및 binning
##############################################
def rolling_minmax_scale(series, window=24):
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

##############################################
# Diffusion 모델을 위한 Dataset 정의 (회귀 + 방향 평가용)
##############################################
class DiffusionTimeSeriesDataset(Dataset):
    def __init__(self, input_data, target_data, lookback=24):
        self.input_data = input_data.values
        self.target_data = target_data.values  # continuous target 값 (scaled close)
        self.lookback = lookback

    def __len__(self):
        return len(self.input_data) - self.lookback

    def __getitem__(self, idx):
        # 조건 데이터: lookback window
        x = self.input_data[idx: idx + self.lookback, :]
        # 타깃: 다음 시점의 target (continuous)
        target = self.target_data[idx + self.lookback, 0]
        # 이전 시점의 target (분류 평가에 사용)
        prev_target = self.target_data[idx + self.lookback - 1, 0]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor([target], dtype=torch.float32),
                torch.tensor([prev_target], dtype=torch.float32))

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
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)

##############################################
# DiffusionClassifier: 조건부 diffusion 모델 (회귀, 평가 시 방향 분류)
##############################################
class DiffusionClassifier(nn.Module):
    def __init__(self, input_dim, lookback, condition_dim=128, num_timesteps=100, hidden_dim=128):
        super(DiffusionClassifier, self).__init__()
        self.num_timesteps = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        self.condition_encoder = ConditionEncoder(input_dim, lookback, condition_dim)
        self.time_embedding = nn.Embedding(num_timesteps, hidden_dim)
        self.model = nn.Sequential(
            nn.Linear(1 + condition_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_condition, y_noisy, t):
        cond = self.condition_encoder(x_condition)
        t_emb = self.time_embedding(t)
        inp = torch.cat([y_noisy, cond, t_emb], dim=1)
        predicted_noise = self.model(inp)
        return predicted_noise
    
    def sample(self, x_condition, device):
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

##############################################
# Diffusion 모델 학습 및 평가 함수 (회귀 + 방향 평가)
##############################################
def train_diffusion_model(model, dataloader, num_epochs, device, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
            alphas_cumprod_t = model.alphas_cumprod[t].view(batch_size, 1)
            noise = torch.randn_like(y)
            y_noisy = torch.sqrt(alphas_cumprod_t) * y + torch.sqrt(1 - alphas_cumprod_t) * noise
            optimizer.zero_grad()
            predicted_noise = model(x, y_noisy, t)
            loss = mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}")

def evaluate_diffusion_model(model, dataloader, device):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for x, y, y_prev in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_prev = y_prev.to(device)
            y_sampled = model.sample(x, device)
            loss = mse_loss(y_sampled, y)
            total_mse += loss.item() * y.size(0)
            total_samples += y.size(0)
            # 예측된 방향: 1 if 예측값 > y_prev, 0 otherwise
            y_pred_class = (y_sampled > y_prev).float()
            # 실제 방향: 1 if y > y_prev, else 0
            y_true_class = (y > y_prev).float()
            correct += (y_pred_class == y_true_class).sum().item()
    avg_mse = total_mse / total_samples
    accuracy = correct / total_samples
    print(f"Evaluation MSE: {avg_mse:.6f}, Accuracy: {accuracy:.4f}")
    return avg_mse, accuracy

##############################################
# 데이터 로드 및 전처리 (OHLC 값 사용: 원본 값에 scaling 후 인코딩)
##############################################
data = pd.read_csv("ETH_upbit_KRW_min5_0309.csv", index_col=0)
data.index = pd.to_datetime(data.index)
data = data[['open', 'high', 'low', 'close']]

# 각 OHLC에 대해 rolling minmax scaling 적용 후 새 컬럼 생성 (scaled 값)
for feature in ['open', 'high', 'low', 'close']:
    data[feature + '_scaled'] = rolling_minmax_scale(data[feature], window=24)
data = data.dropna()

# one-hot 인코딩: _scaled 컬럼 대상 (각 100구간 → 총 400차원)
features_to_bin = ['open_scaled', 'high_scaled', 'low_scaled', 'close_scaled']
data = bin_and_encode(data, features_to_bin, bins=100, drop_original=True)

# 타깃: close_scaled 컬럼을 그대로 사용 (continuous regression target)
data['close_target'] = data['close_scaled']
data = data.dropna()

# 최종 입력: '_scaled_Bin_'가 포함된 열들만 선택
final_input_columns = [col for col in data.columns if '_scaled_Bin_' in col]
final_target_column = ['close_target']

data_input = data[final_input_columns]
data_target = data[final_target_column]

##############################################
# 실험 실행: Diffusion Model 기반 주가 예측 (회귀 + 방향 평가)
##############################################
def train_and_evaluate_diffusion(data, num_experiments=16, lookback=24, num_epochs=10):
    final_input_columns = [col for col in data.columns if '_scaled_Bin_' in col]
    target_cols = ['close_target']
    
    data_input = data[final_input_columns]
    data_target = data[target_cols]
    
    data_input = data_input.apply(pd.to_numeric).astype(np.float32)
    data_target = data_target.apply(pd.to_numeric).astype(np.float32)
    
    step_size = 31200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    val_mse_list = []
    test_mse_list = []
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
        val_mse, val_acc = evaluate_diffusion_model(model, val_loader, device)
        val_mse_list.append(val_mse)
        val_acc_list.append(val_acc)
        
        print("Test Evaluation:")
        test_mse, test_acc = evaluate_diffusion_model(model, test_loader, device)
        test_mse_list.append(test_mse)
        test_acc_list.append(test_acc)
    
        print(f"Experiment {exp}: Validation MSE: {val_mse:.6f}, val_Accuracy: {val_acc:.4f}, test_Accuracy: {test_acc:.4f}")
    
    if val_mse_list:
        avg_val_mse = sum(val_mse_list) / len(val_mse_list)
        avg_test_mse = sum(test_mse_list) / len(test_mse_list)
        avg_val_acc = sum(val_acc_list) / len(val_acc_list)
        avg_test_acc = sum(test_acc_list) / len(test_acc_list)
        print(f"\nFinal Average Validation MSE: {avg_val_mse:.6f}")
        print(f"Final Average Test MSE: {avg_test_mse:.6f}")
        print(f"Final Average Val Accuracy: {avg_val_acc:.4f}")
        print(f"Final Average Test Accuracy: {avg_test_acc:.4f}")
    else:
        print("실험이 한 번도 실행되지 않았습니다.")

##############################################
# 전체 실행 시간 측정
##############################################
start_time = time.time()
train_and_evaluate_diffusion(data, num_experiments=16, lookback=24, num_epochs=10)
end_time = time.time()
elapsed = end_time - start_time
print(f"\n총 수행 시간: {elapsed:.2f}초")
