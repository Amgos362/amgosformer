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
        self.feature_means = None
        self.feature_stds = None
        
    def rolling_normalize(self, series):
        """Rolling normalization with memory of statistics"""
        roll_mean = series.rolling(window=self.window, min_periods=1).mean()
        roll_std = series.rolling(window=self.window, min_periods=1).std()
        normalized = (series - roll_mean) / (roll_std + 1e-8)
        return normalized.fillna(0)
    
    def create_technical_features(self, data):
        """Create additional technical features"""
        df = data.copy()
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['log_return'].rolling(window=self.window).std()
        
        # Price ranges
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Moving averages
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def encode_ohlc(self, data):
        """Enhanced OHLC encoding"""
        df = data.copy()
        
        # 1. Normalize OHLC
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            df[f'{col}_norm'] = self.rolling_normalize(df[col])
        
        # 2. Create price position features
        df['high_pos'] = (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['low_pos'] = (df['low'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['close_pos'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        
        # 3. Add technical features
        df = self.create_technical_features(df)
        
        # 4. Time features
        df['hour'] = pd.to_datetime(df.index).hour / 24.0
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek / 7.0
        
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
        # Ensure correct shapes for batch_first=True
        if len(q.shape) == 2:
            q = q.unsqueeze(1)  # [batch, 1, d_model]
        if len(k.shape) == 2:
            k = k.unsqueeze(1)  # [batch, 1, d_model]
        if len(v.shape) == 2:
            v = v.unsqueeze(1)  # [batch, 1, d_model]
            
        # Ensure all tensors have the same batch size
        batch_size = q.size(0)
        if k.size(0) != batch_size:
            k = k.expand(batch_size, -1, -1)
        if v.size(0) != batch_size:
            v = v.expand(batch_size, -1, -1)
            
        attn_output, _ = self.mha(q, k, v, key_padding_mask=mask)
        return self.norm(attn_output + q)

class CSDIScoreNetwork(nn.Module):
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
        # Ensure correct shapes
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, d_model]
        if len(context.shape) == 2:
            context = context.unsqueeze(1)  # [batch, 1, d_model]
            
        # Self attention on target
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        
        # Cross attention with context
        x = self.norm2(x)
        x = self.cross_attn(x, context, context)
        
        # Feed forward
        x = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x))
        return x

class CSDIDiffusion(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=8, num_layers=3, num_timesteps=100):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        
        # Beta schedule
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
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
            CSDIScoreNetwork(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.final_proj = nn.Linear(d_model, 1)
    
    def get_diffusion_time_embeddings(self, t, x):
        batch_size = x.size(0)
        t_emb = self.time_embed(t)  # [batch_size, d_model]
        return t_emb.unsqueeze(1)  # [batch_size, 1, d_model]
        
    def forward(self, x_condition, y_noisy, t):
        batch_size = x_condition.size(0)
        
        # Project and encode input (shape: [batch, lookback, d_model])
        x = self.input_proj(x_condition)
        x = self.pos_encoder(x)
        
        # Project y_noisy to d_model dimensions
        # Ensure y_noisy has shape [batch_size, 1]
        if len(y_noisy.shape) == 1:
            y_noisy = y_noisy.unsqueeze(-1)
        y = self.target_proj(y_noisy)  # [batch_size, 1, d_model]
        
        # Time embedding
        t_emb = self.get_diffusion_time_embeddings(t, x_condition)
        
        # Add time embedding to condition
        x = x + t_emb
        
        # Process through score networks
        for score_net in self.score_networks:
            y = score_net(y, x)
            
        # Final projection back to scalar
        return self.final_proj(y).squeeze(-1)  # [batch_size]
    
    def sample(self, x_condition, device, num_samples=1):
        batch_size = x_condition.size(0)
        # Initialize y with shape [batch_size, 1]
        y = torch.randn(batch_size, 1, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self.forward(x_condition, y, t_tensor)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # DDPM sampling step
            y = (1 / torch.sqrt(alpha)) * (y - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta) * noise
                
        return y.squeeze(-1)  # Return shape: [batch_size]

class CSDIScoreNetwork(nn.Module):
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
        # Self attention on target
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        
        # Cross attention with context
        x = self.norm2(x)
        x = self.cross_attn(x, context, context)
        
        # Feed forward
        x = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x))
        return x

class CSDITimeSeriesDataset(Dataset):
    def __init__(self, input_data, target_data, lookback=24):
        self.input_data = input_data.values
        self.target_data = target_data.values
        self.lookback = lookback

    def __len__(self):
        return len(self.input_data) - self.lookback

    def __getitem__(self, idx):
        x = self.input_data[idx:idx + self.lookback]  # Shape: [lookback, features]
        target = self.target_data[idx + self.lookback]  # Shape: scalar
        prev_target = self.target_data[idx + self.lookback - 1]  # Shape: scalar
        return (
            torch.tensor(x, dtype=torch.float32),  # Shape: [lookback, features]
            torch.tensor([target], dtype=torch.float32),  # Shape: [1]
            torch.tensor([prev_target], dtype=torch.float32)  # Shape: [1]
        )

def prepare_csdi_data(data_path: str, lookback: int = 24) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for CSDI model"""
    # Load data
    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Initialize preprocessor
    preprocessor = ImprovedDataPreprocessing(window=lookback)
    
    # Process data
    processed_data = preprocessor.encode_ohlc(data)
    
    # Select features
    feature_columns = [
        'open_norm', 'high_norm', 'low_norm', 'close_norm',
        'high_pos', 'low_pos', 'close_pos',
        'log_return', 'volatility', 'price_range',
        'ma7', 'ma21', 'rsi',
        'hour', 'day_of_week'
    ]
    
    # Prepare input and target
    X = processed_data[feature_columns].fillna(0)
    y = processed_data['close']  # Original close price as target
    
    return X, y

def prepare_train_val_test_split(X: pd.DataFrame, y: pd.Series, 
                               train_ratio: float = 0.7, 
                               val_ratio: float = 0.15) -> Tuple[Tuple[pd.DataFrame, pd.Series], 
                                                               Tuple[pd.DataFrame, pd.Series], 
                                                               Tuple[pd.DataFrame, pd.Series]]:
    """Split data into train, validation, and test sets"""
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Train set
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    # Validation set
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    # Test set
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"Data split sizes:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_csdi_model(model, train_loader, val_loader, num_epochs, device, patience=5, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Handle batch data correctly
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # First element is input features
                y = batch[1]  # Second element is target
            else:
                x = batch['x']  # If batch is a dictionary
                y = batch['y']
                
            x = x.to(device)
            y = y.to(device)
            
            # Generate random timesteps
            t = torch.randint(0, model.num_timesteps, (x.size(0),), device=device).long()
            
            # Add noise to target
            noise = torch.randn_like(y)
            y_noisy = torch.sqrt(model.alphas_cumprod[t].view(-1, 1)) * y + \
                     torch.sqrt(1 - model.alphas_cumprod[t].view(-1, 1)) * noise
            
            # Forward pass
            predicted_noise = model(x, y_noisy, t)
            
            # Ensure shapes match for loss calculation
            predicted_noise = predicted_noise.view(-1, 1)  # [batch_size, 1]
            noise = noise.view(-1, 1)  # [batch_size, 1]
            
            loss = criterion(predicted_noise, noise)
            
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
                # Handle batch data correctly
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1]
                else:
                    x = batch['x']
                    y = batch['y']
                    
                x = x.to(device)
                y = y.to(device)
                
                t = torch.randint(0, model.num_timesteps, (x.size(0),), device=device).long()
                noise = torch.randn_like(y)
                y_noisy = torch.sqrt(model.alphas_cumprod[t].view(-1, 1)) * y + \
                         torch.sqrt(1 - model.alphas_cumprod[t].view(-1, 1)) * noise
                
                predicted_noise = model(x, y_noisy, t)
                
                # Ensure shapes match for loss calculation
                predicted_noise = predicted_noise.view(-1, 1)
                noise = noise.view(-1, 1)
                
                loss = criterion(predicted_noise, noise)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_csdi_model(model: nn.Module, 
                       dataloader: DataLoader, 
                       device: str) -> Tuple[float, float, List[float], List[float]]:
    """Evaluate CSDI model"""
    model.eval()
    predictions = []
    true_values = []
    total_mse = 0
    correct_directions = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y, y_prev in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_prev = y_prev.to(device)
            
            y_pred = model.sample(x, device)
            
            predictions.extend(y_pred.cpu().numpy().flatten())
            true_values.extend(y.cpu().numpy().flatten())
            
            mse = ((y_pred - y) ** 2).mean()
            total_mse += mse.item() * y.size(0)
            
            pred_direction = (y_pred > y_prev).float()
            true_direction = (y > y_prev).float()
            correct_directions += (pred_direction == true_direction).sum().item()
            
            total_samples += y.size(0)
    
    avg_mse = total_mse / total_samples
    direction_accuracy = correct_directions / total_samples
    
    return avg_mse, direction_accuracy, predictions, true_values

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Split data
    train_data, val_data, test_data = split_data(data)
    
    # Create datasets
    train_dataset = CryptoDataset(train_data)
    val_dataset = CryptoDataset(val_data)
    test_dataset = CryptoDataset(test_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Print data split sizes
    print("Data split sizes:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = CSDIDiffusion(
        input_dim=15,  # Number of features
        d_model=128,
        num_heads=8,
        num_layers=3,
        num_timesteps=100
    ).to(device)
    
    # Check batch structure
    print("\nChecking batch structure:")
    for batch in train_loader:
        print("Batch type:", type(batch))
        if isinstance(batch, (list, tuple)):
            print("Batch length:", len(batch))
            print("Batch[0] shape:", batch[0].shape)
            print("Batch[1] shape:", batch[1].shape)
        else:
            print("Batch keys:", batch.keys())
        break  # Only check first batch
    
    # Train model
    model = train_csdi_model(
        model, 
        train_loader,
        val_loader,
        NUM_EPOCHS,
        device,
        patience=PATIENCE,
        lr=LEARNING_RATE
    )
    
    # Save model
    torch.save(model.state_dict(), 'csdi_model_final.pth')
    print("\nModel saved as 'csdi_model_final.pth'")

if __name__ == "__main__":
    main()