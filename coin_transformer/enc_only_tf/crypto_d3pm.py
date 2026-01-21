import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import time
from typing import Tuple, List

class ImprovedDataPreprocessing:
    def __init__(self, window=24):
        self.window = window
        
    def rolling_normalize(self, series):
        """Rolling normalization with memory of statistics"""
        roll_mean = series.rolling(window=self.window, min_periods=1).mean()
        roll_std = series.rolling(window=self.window, min_periods=1).std()
        normalized = (series - roll_mean) / (roll_std + 1e-8)
        return normalized.fillna(0)
    
    def rolling_minmax_scale(self, series):
        """Rolling min-max scaling"""
        roll_min = series.rolling(window=self.window, min_periods=1).min()
        roll_max = series.rolling(window=self.window, min_periods=1).max()
        scaled = (series - roll_min) / (roll_max - roll_min + 1e-8)
        return scaled.fillna(0).clip(0, 1)
    
    def bin_and_encode(self, data, features, bins=100):
        """Bin and one-hot encode features"""
        result_df = data.copy()
        
        for feature in features:
            # Create bins
            result_df[f'{feature}_Bin'] = pd.cut(result_df[feature], bins=bins, labels=False)
            # Create one-hot encoding
            one_hot = pd.get_dummies(result_df[f'{feature}_Bin'], prefix=f'{feature}_Bin')
            # Ensure all bin columns exist
            expected_columns = [f'{feature}_Bin_{i}' for i in range(bins)]
            one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
            # Add one-hot columns to result
            result_df = pd.concat([result_df, one_hot], axis=1)
            # Drop the intermediate bin column
            result_df = result_df.drop(columns=[f'{feature}_Bin'])
        
        return result_df
    
    def encode_ohlc(self, data):
        """Enhanced OHLC encoding with binning"""
        df = data.copy()
        
        # MinMax scale OHLC
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            df[col] = self.rolling_minmax_scale(df[col])
        
        # Bin and encode OHLC
        df = self.bin_and_encode(df, ohlc_cols, bins=100)
        
        return df

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, q, k, v, mask=None):
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        if len(k.shape) == 2:
            k = k.unsqueeze(1)
        if len(v.shape) == 2:
            v = v.unsqueeze(1)
            
        batch_size = q.size(0)
        if k.size(0) != batch_size:
            k = k.expand(batch_size, -1, -1)
        if v.size(0) != batch_size:
            v = v.expand(batch_size, -1, -1)
            
        attn_output, _ = self.mha(q, k, v, key_padding_mask=mask)
        return self.norm(attn_output + q)

class D3PMScoreNetwork(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=512):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, context):
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        
        x = self.norm2(x)
        x = self.cross_attn(x, context, context)
        
        x = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x))
        return x

class D3PMDiffusion(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=8, num_layers=3, num_timesteps=100):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        
        # D3PM transition matrix parameters
        self.num_classes = 2  # Binary classification
        self.Q = self._get_transition_matrix()
        self.register_buffer('Qt', self._get_transition_matrices())
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Target projection
        self.target_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(num_timesteps, d_model),
            nn.Linear(d_model, d_model)
        )
        
        # Score networks
        self.score_networks = nn.ModuleList([
            D3PMScoreNetwork(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        # Final projection (binary classification)
        self.final_proj = nn.Linear(d_model, 2)
    
    def _get_transition_matrix(self):
        """Get D3PM transition matrix Q"""
        Q = torch.zeros(2, 2)  # Binary classification
        Q[0, 1] = 0.1  # Probability of 0 -> 1
        Q[1, 0] = 0.1  # Probability of 1 -> 0
        Q[0, 0] = 1 - Q[0, 1]
        Q[1, 1] = 1 - Q[1, 0]
        return Q
    
    def _get_transition_matrices(self):
        """Get time-dependent transition matrices Qt"""
        Qt = []
        Q = self.Q
        for t in range(self.num_timesteps):
            beta_t = (t + 1) / self.num_timesteps
            Qt_t = torch.matrix_power(Q, int(beta_t * 100))
            Qt.append(Qt_t)
        return torch.stack(Qt)
    
    def add_noise(self, x, t):
        """Add D3PM-style discrete noise"""
        batch_size = x.size(0)
        Qt_t = self.Qt[t]  # [batch_size, 2, 2]
        
        # Ensure Qt_t is properly broadcasted for each sample in the batch
        if len(t) > 1:  # If we have a batch of different timesteps
            Qt_t = Qt_t.to(x.device)  # Ensure same device
        
        # Convert to one-hot
        x_onehot = torch.zeros(batch_size, 2, device=x.device)
        x_onehot.scatter_(1, x.long(), 1)  # [batch_size, 2]
        
        # Apply transition matrix for each sample in the batch
        probs = []
        for i in range(batch_size):
            if len(t) > 1:
                curr_Qt = Qt_t[i]  # Get transition matrix for current timestep
            else:
                curr_Qt = Qt_t
            # Compute probability for current sample
            prob = torch.matmul(x_onehot[i], curr_Qt)  # [2]
            probs.append(prob)
        
        # Stack probabilities
        probs = torch.stack(probs)  # [batch_size, 2]
        
        # Ensure probabilities are valid
        probs = torch.clamp(probs, min=1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample new states
        try:
            noisy_x = torch.multinomial(probs, 1)
        except RuntimeError as e:
            print(f"Probs shape: {probs.shape}")
            print(f"Probs sum: {probs.sum(dim=-1)}")
            print(f"Probs min: {probs.min()}, max: {probs.max()}")
            raise e
        
        return noisy_x
    
    def get_diffusion_time_embeddings(self, t, x):
        batch_size = x.size(0)
        t_emb = self.time_embed(t)
        return t_emb.unsqueeze(1)
        
    def forward(self, x_condition, y_noisy, t):
        batch_size = x_condition.size(0)
        
        # Project and encode input
        x = self.input_proj(x_condition)
        x = self.pos_encoder(x)
        
        # Project y_noisy
        if len(y_noisy.shape) == 1:
            y_noisy = y_noisy.unsqueeze(-1)
        y = self.target_proj(y_noisy.float())
        
        # Time embedding
        t_emb = self.get_diffusion_time_embeddings(t, x_condition)
        
        # Add time embedding to condition
        x = x + t_emb
        
        # Process through score networks
        for score_net in self.score_networks:
            y = score_net(y, x)
            
        # Final projection to logits
        return self.final_proj(y.squeeze(1))
    
    def sample(self, x_condition, device):
        batch_size = x_condition.size(0)
        # Start with random binary states
        y = torch.randint(0, 2, (batch_size, 1), device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            logits = self.forward(x_condition, y, t_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            # Ensure probabilities are valid
            probs = torch.clamp(probs, min=1e-6)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            y = torch.multinomial(probs, 1)
            
        return y

class D3PMTimeSeriesDataset(Dataset):
    def __init__(self, input_data, target_data, lookback=24):
        self.input_data = input_data.values
        self.target_data = target_data.values
        self.lookback = lookback

    def __len__(self):
        return len(self.input_data) - self.lookback

    def __getitem__(self, idx):
        x = self.input_data[idx:idx + self.lookback]
        target = self.target_data[idx + self.lookback]
        prev_target = self.target_data[idx + self.lookback - 1]
        
        # Convert target to binary (0: down, 1: up)
        binary_target = float(target > prev_target)
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([binary_target], dtype=torch.long),
            torch.tensor([prev_target], dtype=torch.float32)
        )

def prepare_d3pm_data(data_path: str, lookback: int = 24) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for D3PM model with binning"""
    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    preprocessor = ImprovedDataPreprocessing(window=lookback)
    processed_data = preprocessor.encode_ohlc(data)
    
    # Select binned features
    feature_columns = [col for col in processed_data.columns if '_Bin_' in col]
    
    X = processed_data[feature_columns].fillna(0)
    y = processed_data['close']  # Original close price for target
    
    return X, y

def prepare_rolling_train_val_test_split(X: pd.DataFrame, y: pd.Series, 
                                       window_size: int = 25000,
                                       step_size: int = 2500,
                                       train_ratio: float = 0.7, 
                                       val_ratio: float = 0.15) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], 
                                                                            Tuple[pd.DataFrame, pd.Series], 
                                                                            Tuple[pd.DataFrame, pd.Series]]]:
    """Split data into multiple train, validation, and test sets using rolling window"""
    splits = []
    total_size = len(X)
    num_windows = (total_size - window_size) // step_size + 1
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > total_size:
            break
            
        X_window = X[start_idx:end_idx]
        y_window = y[start_idx:end_idx]
        
        # Split window into train/val/test
        train_size = int(window_size * train_ratio)
        val_size = int(window_size * val_ratio)
        
        X_train = X_window[:train_size]
        y_train = y_window[:train_size]
        
        X_val = X_window[train_size:train_size + val_size]
        y_val = y_window[train_size:train_size + val_size]
        
        X_test = X_window[train_size + val_size:]
        y_test = y_window[train_size + val_size:]
        
        splits.append(((X_train, y_train), (X_val, y_val), (X_test, y_test)))
        
    return splits

def train_d3pm_model(model, train_loader, val_loader, num_epochs, device, patience=5, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            t = torch.randint(0, model.num_timesteps, (x.size(0),), device=device).long()
            
            # Add D3PM noise
            y_noisy = model.add_noise(y, t)
            
            # Forward pass
            logits = model(x, y_noisy, t)
            
            loss = criterion(logits, y.squeeze(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                
                t = torch.randint(0, model.num_timesteps, (x.size(0),), device=device).long()
                y_noisy = model.add_noise(y, t)
                
                logits = model(x, y_noisy, t)
                loss = criterion(logits, y.squeeze(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_d3pm_model(model: nn.Module, 
                       dataloader: DataLoader, 
                       device: str) -> Tuple[float, float, List[float], List[float]]:
    """Evaluate D3PM model"""
    model.eval()
    predictions = []
    true_values = []
    total_loss = 0
    correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y, y_prev in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model.sample(x, device)
            
            predictions.extend(y_pred.cpu().numpy().flatten())
            true_values.extend(y.cpu().numpy().flatten())
            
            # Calculate cross-entropy loss
            logits = model(x, y_pred, torch.zeros(x.size(0), device=device).long())
            loss = criterion(logits, y.squeeze(-1))
            total_loss += loss.item() * y.size(0)
            
            # Calculate accuracy
            pred_labels = torch.argmax(logits, dim=1)
            correct += (pred_labels == y.squeeze(-1)).sum().item()
            
            total_samples += y.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    
    return avg_loss, accuracy, predictions, true_values

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    PATIENCE = 5
    LEARNING_RATE = 1e-4
    WINDOW_SIZE = 25000
    STEP_SIZE = 2500
    
    # Load and preprocess data
    X, y = prepare_d3pm_data('BTC_upbit_KRW_min60.csv')
    
    # Get rolling splits
    splits = prepare_rolling_train_val_test_split(
        X, y,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE
    )
    
    print(f"Total number of rolling windows: {len(splits)}")
    
    # Store accuracies for each window
    window_accuracies = []
    
    # Train and evaluate for each window
    for window_idx, ((X_train, y_train), (X_val, y_val), (X_test, y_test)) in enumerate(splits):
        print(f"\nTraining on window {window_idx + 1}/{len(splits)}")
        print(f"Window range: {window_idx * STEP_SIZE} to {window_idx * STEP_SIZE + WINDOW_SIZE}")
        
        # Create datasets
        train_dataset = D3PMTimeSeriesDataset(X_train, y_train)
        val_dataset = D3PMTimeSeriesDataset(X_val, y_val)
        test_dataset = D3PMTimeSeriesDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = D3PMDiffusion(
            input_dim=400,  # Updated for binned features (4 * 100)
            d_model=128,
            num_heads=8,
            num_layers=3,
            num_timesteps=100
        ).to(device)
        
        # Load previous window's model if available
        if window_idx > 0:
            prev_model_path = f'd3pm_model_window_{window_idx}.pth'
            try:
                model.load_state_dict(torch.load(prev_model_path))
                print(f"Loaded model from window {window_idx} for initialization")
            except FileNotFoundError:
                print(f"No previous model found at {prev_model_path}, starting with fresh initialization")
        
        # Train model
        model = train_d3pm_model(
            model, 
            train_loader,
            val_loader,
            NUM_EPOCHS,
            device,
            patience=PATIENCE,
            lr=LEARNING_RATE
        )
        
        # Evaluate model
        test_loss, test_accuracy, predictions, true_values = evaluate_d3pm_model(
            model,
            test_loader,
            device
        )
        
        print(f"\nWindow {window_idx + 1} Test Results:")
        print(f"Loss: {test_loss:.6f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        
        window_accuracies.append(test_accuracy)
        
        # Save model for this window
        torch.save(model.state_dict(), f'd3pm_model_window_{window_idx + 1}.pth')
    
    # Calculate and print average accuracy across all windows
    avg_accuracy = sum(window_accuracies) / len(window_accuracies)
    print(f"\nFinal Results:")
    print(f"Average accuracy across all windows: {avg_accuracy:.4f}")
    print(f"Individual window accuracies: {[f'{acc:.4f}' for acc in window_accuracies]}")

if __name__ == "__main__":
    main() 