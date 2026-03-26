"""
ChartViT-Distribution Model
============================
Combines two ideas:
1. Chart images (candlestick + indicators) as ViT input
2. Probability distribution output over discretized price levels

Multi-timeframe architecture with cross-attention fusion.

Usage:
    python chart_vit_distribution.py --data_dir ../data --timeframes 5m 1h 1d
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO
from PIL import Image
import math
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. Chart Image Generation
# ============================================================

class ChartImageGenerator:
    """
    OHLCV + 기술적 지표 → 캔들스틱 차트 이미지 생성
    각 이미지는 lookback 봉의 차트를 담음
    """
    def __init__(self, img_size=224, lookback=64, indicators=True):
        self.img_size = img_size
        self.lookback = lookback
        self.indicators = indicators
        # mplfinance style (dark background, clean)
        self.style = mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mpf.make_marketcolors(
                up='#00cc00', down='#ff3333',
                wick={'up': '#00cc00', 'down': '#ff3333'},
                edge={'up': '#00cc00', 'down': '#ff3333'},
                volume='#4488ff'
            ),
            figcolor='black', facecolor='black',
            gridcolor='#222222', gridstyle='-'
        )

    def generate_chart_image(self, ohlcv_df):
        """
        ohlcv_df: DataFrame with columns [open, high, low, close, volume]
                  and DatetimeIndex, length = self.lookback
        Returns: numpy array of shape (3, img_size, img_size), normalized to [0, 1]
        """
        addplots = []
        if self.indicators and len(ohlcv_df) >= 20:
            # SMA lines
            sma5 = ohlcv_df['close'].rolling(5).mean()
            sma20 = ohlcv_df['close'].rolling(20).mean()
            addplots.append(mpf.make_addplot(sma5, color='#ffff00', width=0.7))
            addplots.append(mpf.make_addplot(sma20, color='#ff00ff', width=0.7))

            # Bollinger Bands
            bb_mid = ohlcv_df['close'].rolling(20).mean()
            bb_std = ohlcv_df['close'].rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            addplots.append(mpf.make_addplot(bb_upper, color='#888888', width=0.5, linestyle='--'))
            addplots.append(mpf.make_addplot(bb_lower, color='#888888', width=0.5, linestyle='--'))

        buf = BytesIO()
        fig, axes = mpf.plot(
            ohlcv_df,
            type='candle',
            style=self.style,
            volume=True,
            addplot=addplots if addplots else None,
            figsize=(3, 3),
            tight_layout=True,
            returnfig=True,
            axisoff=True
        )
        fig.savefig(buf, format='png', dpi=self.img_size // 3,
                    bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf).convert('RGB').resize(
            (self.img_size, self.img_size), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        return arr.transpose(2, 0, 1)  # (3, H, W)

    def generate_chart_tensor_fast(self, ohlcv_df):
        """
        Fast rendering: draw candles directly as numpy array (no matplotlib).
        Much faster for training. Produces simplified chart image.
        """
        H, W = self.img_size, self.img_size
        img = np.zeros((H, W, 3), dtype=np.float32)

        opens = ohlcv_df['open'].values
        highs = ohlcv_df['high'].values
        lows = ohlcv_df['low'].values
        closes = ohlcv_df['close'].values
        volumes = ohlcv_df['volume'].values

        n = len(ohlcv_df)
        price_min = lows.min()
        price_max = highs.max()
        price_range = price_max - price_min + 1e-8
        vol_max = volumes.max() + 1e-8

        # Reserve bottom 20% for volume
        price_area_h = int(H * 0.8)
        vol_area_h = H - price_area_h

        candle_width = max(1, W // (n + 1))
        gap = max(1, (W - candle_width * n) // (n + 1))

        for i in range(n):
            x_center = gap + i * (candle_width + gap) + candle_width // 2
            if x_center >= W:
                break

            # Price → pixel y (inverted: high price = low y)
            def price_to_y(p):
                return int((1.0 - (p - price_min) / price_range) * (price_area_h - 1))

            y_open = price_to_y(opens[i])
            y_close = price_to_y(closes[i])
            y_high = price_to_y(highs[i])
            y_low = price_to_y(lows[i])

            is_up = closes[i] >= opens[i]
            color = np.array([0.0, 0.8, 0.0]) if is_up else np.array([1.0, 0.2, 0.2])

            # Wick (high-low line)
            y_top, y_bot = min(y_high, y_low), max(y_high, y_low)
            y_top = max(0, min(y_top, price_area_h - 1))
            y_bot = max(0, min(y_bot, price_area_h - 1))
            if x_center < W:
                img[y_top:y_bot + 1, x_center, :] = color

            # Body (open-close rectangle)
            body_top = min(y_open, y_close)
            body_bot = max(y_open, y_close)
            body_top = max(0, min(body_top, price_area_h - 1))
            body_bot = max(0, min(body_bot, price_area_h - 1))
            x_left = max(0, x_center - candle_width // 2)
            x_right = min(W - 1, x_center + candle_width // 2)
            img[body_top:body_bot + 1, x_left:x_right + 1, :] = color

            # Volume bar
            vol_h = int((volumes[i] / vol_max) * (vol_area_h - 1))
            vol_y_top = price_area_h + (vol_area_h - vol_h)
            vol_y_top = max(price_area_h, min(vol_y_top, H - 1))
            vol_color = np.array([0.27, 0.53, 1.0])
            img[vol_y_top:H, x_left:x_right + 1, :] = vol_color

        # Add SMA overlay if enough data
        if self.indicators and n >= 20:
            sma5 = pd.Series(closes).rolling(5).mean().values
            sma20 = pd.Series(closes).rolling(20).mean().values
            sma_colors = [
                (sma5, np.array([1.0, 1.0, 0.0])),
                (sma20, np.array([1.0, 0.0, 1.0]))
            ]
            for sma, sc in sma_colors:
                prev_y = None
                for i in range(n):
                    if np.isnan(sma[i]):
                        prev_y = None
                        continue
                    x = gap + i * (candle_width + gap) + candle_width // 2
                    if x >= W:
                        break
                    y = int((1.0 - (sma[i] - price_min) / price_range) * (price_area_h - 1))
                    y = max(0, min(y, price_area_h - 1))
                    if prev_y is not None and x < W:
                        # Draw line from prev to current
                        y0, y1 = prev_y, y
                        steps = max(abs(y1 - y0), 1)
                        for s in range(steps + 1):
                            yy = int(y0 + (y1 - y0) * s / steps)
                            yy = max(0, min(yy, price_area_h - 1))
                            if x < W:
                                img[yy, x, :] = sc * 0.7
                    prev_y = y

        return img.transpose(2, 0, 1)  # (3, H, W)


# ============================================================
# 2. Technical Indicators
# ============================================================

def add_technical_indicators(df):
    """Add common technical indicators to OHLCV DataFrame."""
    d = df.copy()
    # SMA
    for w in [5, 10, 20, 60]:
        d[f'sma_{w}'] = d['close'].rolling(w).mean()
    # RSI
    delta = d['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    # Bollinger Bands width
    bb_mid = d['close'].rolling(20).mean()
    bb_std = d['close'].rolling(20).std()
    d['bb_width'] = (2 * bb_std) / (bb_mid + 1e-8)
    # MACD
    ema12 = d['close'].ewm(span=12).mean()
    ema26 = d['close'].ewm(span=26).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    # Stochastic %K
    low14 = d['low'].rolling(14).min()
    high14 = d['high'].rolling(14).max()
    d['stoch_k'] = (d['close'] - low14) / (high14 - low14 + 1e-8) * 100
    # ATR
    tr1 = d['high'] - d['low']
    tr2 = abs(d['high'] - d['close'].shift(1))
    tr3 = abs(d['low'] - d['close'].shift(1))
    d['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    # Log return
    d['log_return'] = np.log(d['close'] / d['close'].shift(1))
    # Volatility
    d['volatility'] = d['log_return'].rolling(20).std()
    return d


# ============================================================
# 3. Dataset
# ============================================================

class MultiTimeframeChartDataset(Dataset):
    """
    Multi-timeframe chart image dataset.
    For each sample:
      - Generates chart images at each timeframe
      - Extracts numeric features for the auxiliary branch
      - Computes target: discretized price distribution
    """
    def __init__(self, timeframe_dfs, lookbacks, img_size=224,
                 num_price_bins=61, max_pct_range=0.15,
                 fast_render=True):
        """
        Args:
            timeframe_dfs: dict of {tf_name: DataFrame} with OHLCV + DatetimeIndex
            lookbacks: dict of {tf_name: int} - lookback window per timeframe
            num_price_bins: number of discrete price levels for output distribution
            max_pct_range: ±percentage range for price bins (0.15 = ±15%)
            fast_render: use fast numpy rendering instead of mplfinance
        """
        self.timeframe_names = sorted(timeframe_dfs.keys())
        self.lookbacks = lookbacks
        self.num_price_bins = num_price_bins
        self.max_pct_range = max_pct_range
        self.fast_render = fast_render

        self.chart_gen = ChartImageGenerator(img_size=img_size, indicators=True)

        # Align timeframes: use the shortest (highest freq) as base
        # Each sample index corresponds to a timestamp in the base timeframe
        self.base_tf = self.timeframe_names[0]  # e.g., '5m'
        base_df = timeframe_dfs[self.base_tf]
        base_df = add_technical_indicators(base_df)

        self.dfs = {}
        for tf in self.timeframe_names:
            df = add_technical_indicators(timeframe_dfs[tf])
            self.dfs[tf] = df

        # Valid indices: need enough lookback in all timeframes
        max_lb = max(lookbacks.values())
        min_start = max(max_lb, 60)  # 60 for indicator warmup
        self.base_df = base_df
        self.valid_indices = list(range(min_start, len(base_df) - 1))

        # Numeric feature columns (for auxiliary branch)
        self.numeric_cols = [
            'rsi', 'bb_width', 'macd', 'macd_signal',
            'stoch_k', 'atr', 'log_return', 'volatility'
        ]

        # Price bin edges: linearly spaced from -max_pct to +max_pct
        self.bin_edges = np.linspace(-max_pct_range, max_pct_range, num_price_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def __len__(self):
        return len(self.valid_indices)

    def _get_aligned_slice(self, tf, base_timestamp, lookback):
        """Get the lookback-sized slice from timeframe df aligned to base_timestamp."""
        df = self.dfs[tf]
        # Find the last row <= base_timestamp
        mask = df.index <= base_timestamp
        if mask.sum() < lookback:
            return None
        end_idx = mask.sum()
        start_idx = end_idx - lookback
        return df.iloc[start_idx:end_idx]

    def __getitem__(self, idx):
        base_idx = self.valid_indices[idx]
        base_timestamp = self.base_df.index[base_idx]

        images = []
        for tf in self.timeframe_names:
            lookback = self.lookbacks[tf]
            if tf == self.base_tf:
                slc = self.base_df.iloc[base_idx - lookback:base_idx]
            else:
                slc = self._get_aligned_slice(tf, base_timestamp, lookback)
                if slc is None:
                    slc = self.dfs[tf].iloc[:lookback]

            ohlcv = slc[['open', 'high', 'low', 'close', 'volume']].copy()
            if self.fast_render:
                img = self.chart_gen.generate_chart_tensor_fast(ohlcv)
            else:
                img = self.chart_gen.generate_chart_image(ohlcv)
            images.append(torch.tensor(img, dtype=torch.float32))

        # Numeric features from base timeframe
        row = self.base_df.iloc[base_idx]
        numeric_feats = []
        for col in self.numeric_cols:
            val = row.get(col, 0.0)
            numeric_feats.append(0.0 if pd.isna(val) else float(val))
        numeric_tensor = torch.tensor(numeric_feats, dtype=torch.float32)

        # Target: next candle close pct change → soft label distribution
        current_close = self.base_df.iloc[base_idx]['close']
        next_close = self.base_df.iloc[base_idx + 1]['close']
        pct_change = (next_close - current_close) / (current_close + 1e-8)

        # Create soft target distribution (Gaussian centered on actual pct_change)
        target_dist = self._make_soft_target(pct_change)
        # Also provide hard label (bin index) for cross-entropy
        hard_label = np.digitize(pct_change, self.bin_edges) - 1
        hard_label = np.clip(hard_label, 0, self.num_price_bins - 1)

        return {
            'images': torch.stack(images),        # (num_tf, 3, H, W)
            'numeric': numeric_tensor,             # (num_features,)
            'target_dist': torch.tensor(target_dist, dtype=torch.float32),
            'hard_label': torch.tensor(hard_label, dtype=torch.long),
            'pct_change': torch.tensor(pct_change, dtype=torch.float32),
        }

    def _make_soft_target(self, pct_change, sigma=0.005):
        """
        Gaussian soft label centered on actual pct_change.
        sigma controls the spread (0.5% default).
        """
        dist = np.exp(-0.5 * ((self.bin_centers - pct_change) / sigma) ** 2)
        dist /= dist.sum() + 1e-8
        return dist


# ============================================================
# 4. Model Components
# ============================================================

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ChartViTEncoder(nn.Module):
    """
    ViT encoder for a single timeframe chart image.
    Image → patches → transformer encoder → CLS token embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=384, depth=6, num_heads=6, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, embed_dim) CLS token, (B, num_patches+1, embed_dim) all tokens
        """
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, embed_dim)
        return cls_out, x


class CrossTimeframeAttention(nn.Module):
    """
    Cross-attention between timeframe embeddings.
    Lower timeframe queries attend to higher timeframe keys/values.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, context):
        """
        query: (B, 1, embed_dim) - lower timeframe CLS
        context: (B, N, embed_dim) - higher timeframe all tokens
        """
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        h, _ = self.cross_attn(q, kv, kv)
        out = query + h
        out = out + self.ffn(out)
        return out


# ============================================================
# 5. Full Model
# ============================================================

class ChartViTDistributionModel(nn.Module):
    """
    Multi-timeframe Chart ViT → Cross-Attention Fusion → Price Distribution

    Architecture:
    1. Each timeframe image → independent ViT encoder → CLS embedding
    2. Cross-attention: short-term queries attend to long-term context
    3. Fused embedding + numeric features → distribution head
    4. Output: softmax over discretized price bins
    """
    def __init__(self, num_timeframes=3, img_size=224, patch_size=16,
                 embed_dim=384, vit_depth=6, num_heads=6,
                 num_numeric_features=8, num_price_bins=61,
                 dropout=0.1, share_vit_weights=False):
        super().__init__()
        self.num_timeframes = num_timeframes
        self.embed_dim = embed_dim
        self.num_price_bins = num_price_bins

        # ViT encoders (one per timeframe, or shared)
        if share_vit_weights:
            vit = ChartViTEncoder(img_size, patch_size, 3,
                                  embed_dim, vit_depth, num_heads, dropout)
            self.vit_encoders = nn.ModuleList([vit] * num_timeframes)
        else:
            self.vit_encoders = nn.ModuleList([
                ChartViTEncoder(img_size, patch_size, 3,
                                embed_dim, vit_depth, num_heads, dropout)
                for _ in range(num_timeframes)
            ])

        # Timeframe positional embeddings
        self.tf_embed = nn.Parameter(torch.randn(num_timeframes, embed_dim) * 0.02)

        # Cross-timeframe attention layers
        self.cross_attns = nn.ModuleList([
            CrossTimeframeAttention(embed_dim, num_heads=4, dropout=dropout)
            for _ in range(num_timeframes - 1)
        ])

        # Numeric feature projection
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Fusion + distribution head
        fusion_dim = embed_dim * 2  # fused visual + numeric
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_price_bins),
        )

        # Auxiliary direction head (up/down binary, for multi-task learning)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, images, numeric_features):
        """
        images: (B, num_tf, 3, H, W)
        numeric_features: (B, num_numeric_features)
        returns:
            dist_logits: (B, num_price_bins) - logits for price distribution
            direction_logit: (B, 1) - logit for up/down
        """
        B = images.shape[0]

        # Encode each timeframe
        cls_tokens = []
        all_tokens_list = []
        for i in range(self.num_timeframes):
            img_i = images[:, i]  # (B, 3, H, W)
            cls_i, tokens_i = self.vit_encoders[i](img_i)
            cls_i = cls_i + self.tf_embed[i]  # add timeframe embedding
            cls_tokens.append(cls_i)
            all_tokens_list.append(tokens_i)

        # Cross-timeframe attention: short-term attends to long-term
        # Assumption: timeframes sorted short→long (e.g., 5m, 1h, 1d)
        fused = cls_tokens[0].unsqueeze(1)  # start with shortest timeframe
        for i in range(self.num_timeframes - 1):
            context = all_tokens_list[i + 1]  # longer timeframe tokens
            fused = self.cross_attns[i](fused, context)

        fused = fused.squeeze(1)  # (B, embed_dim)

        # Numeric branch
        numeric_emb = self.numeric_proj(numeric_features)  # (B, embed_dim)

        # Combine
        combined = torch.cat([fused, numeric_emb], dim=-1)  # (B, embed_dim*2)

        # Heads
        dist_logits = self.head(combined)          # (B, num_price_bins)
        direction_logit = self.direction_head(combined)  # (B, 1)

        return dist_logits, direction_logit


# ============================================================
# 6. Loss Functions
# ============================================================

class DistributionLoss(nn.Module):
    """
    Combined loss:
    1. KL divergence between predicted and soft target distribution
    2. Cross-entropy with hard bin label
    3. Binary cross-entropy for direction (up/down)
    """
    def __init__(self, kl_weight=1.0, ce_weight=1.0, dir_weight=0.5):
        super().__init__()
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.dir_weight = dir_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, dist_logits, direction_logit, target_dist, hard_label, pct_change):
        # KL divergence: pred vs soft target
        log_pred = F.log_softmax(dist_logits, dim=-1)
        kl = F.kl_div(log_pred, target_dist, reduction='batchmean')

        # Cross-entropy with hard label
        ce = self.ce_loss(dist_logits, hard_label)

        # Direction loss
        direction_target = (pct_change > 0).float().unsqueeze(-1)
        dir_loss = self.bce_loss(direction_logit, direction_target)

        total = self.kl_weight * kl + self.ce_weight * ce + self.dir_weight * dir_loss
        return total, {
            'kl': kl.item(),
            'ce': ce.item(),
            'dir': dir_loss.item(),
            'total': total.item()
        }


# ============================================================
# 7. Training
# ============================================================

def create_synthetic_data(num_rows=10000, freq='5min'):
    """Create synthetic OHLCV data for testing the pipeline."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=num_rows, freq=freq)
    close = 50000 + np.cumsum(np.random.randn(num_rows) * 100)
    high = close + np.abs(np.random.randn(num_rows) * 50)
    low = close - np.abs(np.random.randn(num_rows) * 50)
    open_ = close + np.random.randn(num_rows) * 30
    volume = np.abs(np.random.randn(num_rows) * 1000) + 500

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume
    }, index=dates)
    return df


def resample_to_timeframe(df, rule):
    """Resample OHLCV data to a higher timeframe."""
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    metrics = {'kl': 0, 'ce': 0, 'dir': 0}
    correct_dir = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['images'].to(device)
        numeric = batch['numeric'].to(device)
        target_dist = batch['target_dist'].to(device)
        hard_label = batch['hard_label'].to(device)
        pct_change = batch['pct_change'].to(device)

        optimizer.zero_grad()
        dist_logits, dir_logit = model(images, numeric)
        loss, loss_dict = criterion(dist_logits, dir_logit, target_dist,
                                     hard_label, pct_change)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_dict['total']
        for k in metrics:
            metrics[k] += loss_dict[k]

        # Direction accuracy
        pred_dir = (dir_logit.squeeze(-1) > 0).float()
        actual_dir = (pct_change > 0).float()
        correct_dir += (pred_dir == actual_dir).sum().item()
        total_samples += len(pct_change)

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'dir_acc': f"{correct_dir / total_samples:.3f}"
        })

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'kl': metrics['kl'] / n,
        'ce': metrics['ce'] / n,
        'dir': metrics['dir'] / n,
        'dir_acc': correct_dir / total_samples
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_dir = 0
    total_samples = 0
    all_pred_bins = []
    all_true_bins = []

    for batch in dataloader:
        images = batch['images'].to(device)
        numeric = batch['numeric'].to(device)
        target_dist = batch['target_dist'].to(device)
        hard_label = batch['hard_label'].to(device)
        pct_change = batch['pct_change'].to(device)

        dist_logits, dir_logit = model(images, numeric)
        loss, loss_dict = criterion(dist_logits, dir_logit, target_dist,
                                     hard_label, pct_change)
        total_loss += loss_dict['total']

        pred_dir = (dir_logit.squeeze(-1) > 0).float()
        actual_dir = (pct_change > 0).float()
        correct_dir += (pred_dir == actual_dir).sum().item()
        total_samples += len(pct_change)

        pred_bins = dist_logits.argmax(dim=-1).cpu().numpy()
        all_pred_bins.extend(pred_bins)
        all_true_bins.extend(hard_label.cpu().numpy())

    n = len(dataloader)
    # Bin prediction accuracy (within ±1 bin)
    all_pred_bins = np.array(all_pred_bins)
    all_true_bins = np.array(all_true_bins)
    bin_acc_exact = (all_pred_bins == all_true_bins).mean()
    bin_acc_near = (np.abs(all_pred_bins - all_true_bins) <= 2).mean()

    return {
        'loss': total_loss / n,
        'dir_acc': correct_dir / total_samples,
        'bin_acc_exact': bin_acc_exact,
        'bin_acc_near': bin_acc_near
    }


def plot_training_history(history, save_path):
    """Generate training summary plots."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ChartViT-Distribution Training Results', fontsize=14, fontweight='bold')

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], 'b-o', markersize=3, label='Train Loss')
    ax.plot(df['epoch'], df['val_loss'], 'r-o', markersize=3, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Direction accuracy
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['train_dir_acc'] * 100, 'b-o', markersize=3, label='Train Dir Acc')
    ax.plot(df['epoch'], df['val_dir_acc'] * 100, 'r-o', markersize=3, label='Val Dir Acc')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Direction Prediction Accuracy (Up/Down)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Bin accuracy
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['val_bin_exact'] * 100, 'g-o', markersize=3, label='Exact Bin Acc')
    ax.plot(df['epoch'], df['val_bin_near'] * 100, 'm-o', markersize=3, label='Near (±2 bins) Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Price Bin Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Loss components
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['train_kl'], '-o', markersize=3, label='KL Div')
    ax.plot(df['epoch'], df['train_ce'], '-o', markersize=3, label='Cross-Entropy')
    ax.plot(df['epoch'], df['train_dir'], '-o', markersize=3, label='Direction BCE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='ChartViT Distribution Model')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing CSV files')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Specific CSV file (overrides data_dir)')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '1h', '1d'],
                        help='Timeframes to use')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--vit_depth', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_price_bins', type=int, default=61,
                        help='Number of price bins (odd number centered on 0)')
    parser.add_argument('--max_pct_range', type=float, default=0.15,
                        help='Max percentage range for price bins (±)')
    parser.add_argument('--lookback', type=int, default=64,
                        help='Base lookback window (for shortest timeframe)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_path', type=str, default='chart_vit_dist.pth')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit dataset size (0=use all)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--toy', action='store_true',
                        help='Quick toy run with small model on real data')
    parser.add_argument('--fast_render', action='store_true', default=True,
                        help='Use fast numpy rendering (default: True)')
    parser.add_argument('--mpl_render', action='store_true',
                        help='Use mplfinance rendering (slow but prettier)')
    args = parser.parse_args()

    if args.mpl_render:
        args.fast_render = False

    # Toy mode: override to small/fast settings
    if args.toy:
        args.img_size = 112
        args.patch_size = 16
        args.embed_dim = 128
        args.vit_depth = 2
        args.num_heads = 4
        args.batch_size = 8
        args.epochs = 3
        args.timeframes = ['1h', '4h']
        args.lookback = 32
        if args.data_file is None:
            args.data_file = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'KRW-ETH_upbit_min60.csv'
            )

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    if args.synthetic:
        print("Using synthetic data for pipeline testing...")
        base_df = create_synthetic_data(num_rows=5000, freq='5min')
    elif args.data_file:
        print(f"Loading data from {args.data_file}")
        base_df = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
        # Normalize column names
        expected = ['open', 'high', 'low', 'close', 'volume']
        base_df.columns = ['open', 'high', 'low', 'close', 'volume', 'value'][:len(base_df.columns)]
        base_df = base_df[expected].dropna()
    else:
        # Try to find a CSV in data_dir
        csv_files = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {args.data_dir}. Use --synthetic for testing.")
            return
        data_path = os.path.join(args.data_dir, csv_files[0])
        print(f"Loading data from {data_path}")
        base_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        expected = ['open', 'high', 'low', 'close', 'volume']
        base_df.columns = ['open', 'high', 'low', 'close', 'volume', 'value'][:len(base_df.columns)]
        base_df = base_df[expected].dropna()

    # Create multi-timeframe data
    tf_resample_map = {
        '1m': '1min', '5m': '5min', '15m': '15min',
        '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'
    }
    lookback_map = {
        '1m': args.lookback, '5m': args.lookback,
        '15m': 48, '30m': 48,
        '1h': 48, '4h': 32, '1d': 32
    }

    timeframe_dfs = {}
    lookbacks = {}
    base_freq = args.timeframes[0]

    for tf in args.timeframes:
        if tf == base_freq:
            timeframe_dfs[tf] = base_df
        else:
            rule = tf_resample_map.get(tf, tf)
            timeframe_dfs[tf] = resample_to_timeframe(base_df, rule)
        lookbacks[tf] = lookback_map.get(tf, args.lookback)
        print(f"  Timeframe {tf}: {len(timeframe_dfs[tf])} rows, lookback={lookbacks[tf]}")

    # Dataset
    dataset = MultiTimeframeChartDataset(
        timeframe_dfs=timeframe_dfs,
        lookbacks=lookbacks,
        img_size=args.img_size,
        num_price_bins=args.num_price_bins,
        max_pct_range=args.max_pct_range,
        fast_render=args.fast_render
    )
    total_samples = len(dataset)
    if args.max_samples > 0 and total_samples > args.max_samples:
        # Take the most recent max_samples (tail of the time series)
        offset = total_samples - args.max_samples
        dataset.valid_indices = dataset.valid_indices[offset:]
        total_samples = len(dataset)
    print(f"Dataset size: {total_samples} samples")

    # Train/val split (time-based, no leakage)
    split_idx = int(len(dataset) * 0.8)
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(dataset)))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    model = ChartViTDistributionModel(
        num_timeframes=len(args.timeframes),
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        vit_depth=args.vit_depth,
        num_heads=args.num_heads,
        num_numeric_features=len(dataset.numeric_cols),
        num_price_bins=args.num_price_bins,
        dropout=0.1,
        share_vit_weights=False,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DistributionLoss(kl_weight=1.0, ce_weight=0.5, dir_weight=0.3)

    # Training loop with logging
    history = []
    best_val_loss = float('inf')
    log_csv = args.save_path.replace('.pth', '_log.csv')

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_metrics['loss'],
            'train_kl': train_metrics['kl'],
            'train_ce': train_metrics['ce'],
            'train_dir': train_metrics['dir'],
            'train_dir_acc': train_metrics['dir_acc'],
            'val_loss': val_metrics['loss'],
            'val_dir_acc': val_metrics['dir_acc'],
            'val_bin_exact': val_metrics['bin_acc_exact'],
            'val_bin_near': val_metrics['bin_acc_near'],
        }
        history.append(row)

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"Train loss={train_metrics['loss']:.4f} dir_acc={train_metrics['dir_acc']:.3f} | "
              f"Val loss={val_metrics['loss']:.4f} dir_acc={val_metrics['dir_acc']:.3f} "
              f"bin_exact={val_metrics['bin_acc_exact']:.3f} bin_near={val_metrics['bin_acc_near']:.3f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'val_metrics': val_metrics,
            }, args.save_path)
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        # Save CSV log incrementally
        pd.DataFrame(history).to_csv(log_csv, index=False)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Log saved to: {log_csv}")

    # Generate training plots
    plot_path = args.save_path.replace('.pth', '_plots.png')
    plot_training_history(history, plot_path)
    print(f"Plots saved to: {plot_path}")


if __name__ == '__main__':
    main()
