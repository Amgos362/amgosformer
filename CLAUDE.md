# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Amgosformer is a collection of encoder-only transformer and diffusion model experiments for cryptocurrency (and stock) price movement prediction. The primary target is ETH on Upbit (Korean exchange), using OHLCV data at various intervals (5min, 60min, daily). The codebase is research-oriented — each Python file is a self-contained experiment, not a library.

## Architecture

All model code lives under `coin_transformer/enc_only_tf/`. There is no shared module system; each file duplicates preprocessing/model definitions with its own variations. The key model families are:

- **Transformer classifiers** (`coinformer*.py`, `cryptoformer*.py`, `coin_transformer.py`, `stock_transformer.py`, `fintransformer.py`): Encoder-only transformers that classify next-candle direction (up/down). Variants differ in feature engineering (technical indicators, datetime encoding, binning strategies) and model architecture details.
- **Diffusion classifiers** (`crypto_diffusion.py`, `cryptodiffusion.py`, `crypto_d3pm.py`, `crypto_csdi_new.py`): Use diffusion processes (DDPM-style noise schedules, D3PM discrete diffusion, CSDI conditional score-based) for the same classification task.
- **Trading scripts** (`ETH_min5_trading_win6.py`, `cryptodiffusion_trading.py`): Live trading bots that load a trained model, fetch real-time data from Upbit via `pyupbit`, and execute buy/sell orders. They send alerts via LINE Notify.
- **Backtesting** (`cryptodiffusion_backtesting.py`): Offline backtesting with `backtrader`.
- **Monitor** (`monitor_script.py`): Watchdog that checks if trading processes are alive and sends LINE notifications.

## Key Patterns

- **Preprocessing pipeline**: OHLCV → technical indicators (SMA, RSI, Bollinger Bands, Stochastic, MACD, etc.) → rolling min-max normalization → binning into 100 bins → one-hot encoding. This produces very high-dimensional input (hundreds of features per candle).
- **Datetime encoding**: Two approaches used across files — cyclical sin/cos encoding (`cryptoformer.py`) or one-hot with learned embeddings (`coinformer_base.py`).
- **Data source**: CSV files (e.g., `KRW-ETH_upbit_min60.csv`) stored in `coin_transformer/data/` (gitignored). Live scripts fetch from Upbit API via `pyupbit`.
- **Framework**: PyTorch. No training framework (no PyTorch Lightning, etc.) — training loops are inline in each script.
- **Language**: Comments and variable names mix Korean and English.

## Running

Each script is standalone. To train a model:
```bash
cd coin_transformer/enc_only_tf
python cryptoformer.py
```

Scripts expect CSV data files in the working directory or `../data/`. Model checkpoints (`.pth`) are saved to the working directory.

## Dependencies

`coin_transformer/enc_only_tf/requirements.txt` lists dependencies. Key packages: `torch`, `pandas`, `numpy`, `pyupbit`, `backtrader`, `pandas_ta`, `mplfinance`, `yfinance`.
