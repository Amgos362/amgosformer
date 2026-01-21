import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

####################################
# 1. Data Preprocessing Functions
####################################
def rolling_minmax_scale(data, window=20, bins=100):
    """
    Rolling MinMax scaling with binning and one-hot encoding
    Args:
        data: pandas Series
        window: rolling window size (20 days)
        bins: number of bins (100 bins)
    Returns:
        one-hot encoded DataFrame
    """
    # Calculate rolling min and max
    roll_min = data.rolling(window=window, min_periods=window).min()
    roll_max = data.rolling(window=window, min_periods=window).max()
    
    # Normalize to [0, 1] with epsilon to avoid division by zero
    scaled = (data - roll_min) / ((roll_max - roll_min) + 1e-8)
    
    # Clip to [0, 1] range
    scaled = scaled.clip(0, 1)
    
    # Convert to bins (0 to bins-1)
    binned = (scaled * (bins - 1)).round().astype(int)
    binned = binned.clip(0, bins - 1)
    
    # One-hot encoding
    one_hot = pd.get_dummies(binned, prefix='bin')
    expected_columns = ['bin_{}'.format(i) for i in range(bins)]
    one_hot = one_hot.reindex(columns=expected_columns, fill_value=0)
    
    return one_hot

def prepare_stock_data(stock_data, lookback=20):
    """
    Prepare data for a single stock
    Args:
        stock_data: DataFrame with OHLCV data for one stock
        lookback: sequence length for transformer
    Returns:
        features, targets, valid_indices
    """
    # Sort by date
    stock_data = stock_data.sort_values('date').reset_index(drop=True)
    
    # Calculate next day return (target)
    stock_data['next_close'] = stock_data['close'].shift(-1)
    stock_data['target'] = (stock_data['next_close'] > stock_data['close']).astype(int)
    
    # Rolling minmax scaling and one-hot encoding for OHLCV
    features_list = []
    for col in ['open', 'high', 'low', 'close', 'volume']:
        one_hot_features = rolling_minmax_scale(stock_data[col], window=20, bins=100)
        # Add column prefix
        one_hot_features.columns = ['{}_{}'.format(col, c) for c in one_hot_features.columns]
        features_list.append(one_hot_features)
    
    # Combine all features
    features = pd.concat(features_list, axis=1)
    
    # Remove rows with NaN (first 20 rows due to rolling window)
    valid_mask = ~(features.isna().any(axis=1) | stock_data['target'].isna())
    features = features[valid_mask].reset_index(drop=True)
    targets = stock_data.loc[valid_mask, 'target'].reset_index(drop=True)
    
    return features.values, targets.values

####################################
# 2. Dataset Definition
####################################
class StockDataset(Dataset):
    def __init__(self, features, targets, lookback=20):
        self.features = features  # shape: (n_samples, n_features)
        self.targets = targets    # shape: (n_samples,)
        self.lookback = lookback
        
    def __len__(self):
        return len(self.features) - self.lookback + 1
    
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.lookback]  # shape: (lookback, n_features)
        # Get target for the last day in sequence
        y = self.targets[idx + self.lookback - 1]   # shape: ()
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

####################################
# 3. Transformer Model Implementation
####################################
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class FinancialTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=8, num_layers=4, ffn_dim=512, num_classes=2, max_seq_len=20):
        super(FinancialTransformer, self).__init__()
        
        # Token embedding (linear projection of input features)
        self.token_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Token embedding
        x = self.token_embedding(x)
        
        # Add positional embedding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.position_embedding(positions)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Use the last token for classification (similar to BERT's [CLS] token)
        x = x[:, -1, :]
        
        # Classification
        output = self.classifier(x)
        
        return output

####################################
# 4. Training and Evaluation Functions
####################################
def calculate_hit_ratio(predictions, targets):
    """Calculate hit ratio (accuracy)"""
    correct = (predictions == targets).sum().item()
    total = len(targets)
    return correct / total if total > 0 else 0.0

def evaluate_model(model, data_loader, device):
    """Evaluate model and return loss and hit ratio"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    hit_ratio = calculate_hit_ratio(np.array(all_predictions), np.array(all_targets))
    
    return avg_loss, hit_ratio

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """Train model for specified epochs"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        train_hit_ratio = calculate_hit_ratio(np.array(all_predictions), np.array(all_targets))
        
        # Validation
        val_loss, val_hit_ratio = evaluate_model(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Hit Ratio: {train_hit_ratio:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Hit Ratio: {val_hit_ratio:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

####################################
# 5. Main Training Loop
####################################
def train_stock_model(stock_code, stock_data, device, lookback=20):
    """Train model for a single stock"""
    print(f"\n=== Training model for stock {stock_code} ===")
    
    # Prepare data
    features, targets = prepare_stock_data(stock_data, lookback=lookback)
    
    if len(features) < 2500:  # Need at least 2500 samples
        print(f"Stock {stock_code} has insufficient data ({len(features)} samples)")
        return None
    
    # Model parameters
    input_dim = features.shape[1]  # 500 features (5 OHLCV * 100 bins each)
    model = FinancialTransformer(
        input_dim=input_dim,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ffn_dim=512,
        num_classes=2,
        max_seq_len=lookback
    ).to(device)
    
    # Training configuration
    epochs = 10
    batch_size = 32
    lr = 1e-4
    
    # Rolling window training
    window_size = 250
    train_size = 2000
    val_size = 250
    test_size = 250
    
    all_test_hit_ratios = []
    model_path = "models/stock_{}_model.pth".format(stock_code)
    os.makedirs("models", exist_ok=True)
    
    # Calculate number of iterations
    max_start_idx = len(features) - train_size - val_size - test_size - lookback + 1
    num_iterations = max(1, max_start_idx // window_size)
    
    for iteration in range(num_iterations):
        start_idx = iteration * window_size
        train_end = start_idx + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        
        if test_end + lookback - 1 >= len(features):
            break
            
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        print(f"Data range: {start_idx} to {test_end}")
        
        # Split data
        train_features = features[start_idx:train_end]
        train_targets = targets[start_idx:train_end]
        val_features = features[train_end:val_end]
        val_targets = targets[train_end:val_end]
        test_features = features[val_end:test_end]
        test_targets = targets[val_end:test_end]
        
        # Create datasets
        train_dataset = StockDataset(train_features, train_targets, lookback)
        val_dataset = StockDataset(val_features, val_targets, lookback)
        test_dataset = StockDataset(test_features, test_targets, lookback)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Load previous model if exists (for continuation training)
        if iteration > 0 and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print("Loaded previous model weights")
        
        # Train model
        model = train_model(model, train_loader, val_loader, epochs, lr, device)
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Evaluate on test set
        test_loss, test_hit_ratio = evaluate_model(model, test_loader, device)
        all_test_hit_ratios.append(test_hit_ratio)
        
        print(f"Test Loss: {test_loss:.4f}, Test Hit Ratio: {test_hit_ratio:.4f}")
    
    if all_test_hit_ratios:
        avg_hit_ratio = np.mean(all_test_hit_ratios)
        print(f"\nAverage Test Hit Ratio for {stock_code}: {avg_hit_ratio:.4f}")
        return avg_hit_ratio
    else:
        return None

def main():
    """Main function to run the training pipeline"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = pd.read_csv("korea_daily_price_data.csv")
    data['date'] = pd.to_datetime(data['date'])
    
    # Get unique stock codes
    stock_codes = data['code'].unique()
    print(f"Found {len(stock_codes)} stocks")
    
    # Train models for each stock (can be parallelized if needed)
    all_hit_ratios = {}
    
    # For demonstration, train on first 10 stocks (remove this limit for full training)
    sample_stocks = stock_codes[:10]  # Remove this line to train on all stocks
    
    for stock_code in tqdm(sample_stocks, desc="Training stocks"):
        stock_data = data[data['code'] == stock_code].copy()
        
        try:
            hit_ratio = train_stock_model(stock_code, stock_data, device)
            if hit_ratio is not None:
                all_hit_ratios[stock_code] = hit_ratio
        except Exception as e:
            print(f"Error training stock {stock_code}: {str(e)}")
            continue
    
    # Print summary
    if all_hit_ratios:
        print("\n=== Training Summary ===")
        for stock_code, hit_ratio in all_hit_ratios.items():
            print(f"Stock {stock_code}: {hit_ratio:.4f}")
        
        overall_avg = np.mean(list(all_hit_ratios.values()))
        print(f"\nOverall Average Hit Ratio: {overall_avg:.4f}")
    else:
        print("No models were successfully trained.")

if __name__ == "__main__":
    main() 