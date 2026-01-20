<div align="center">

# HIEU

### Regime-Aware Hypernetwork Experts for Explainable Multi-Asset Cryptocurrency Forecasting

[![Paper](https://img.shields.io/badge/Paper-IJCAI%202026-blue)](main.pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<img src="plots/HIEU.png" width="800">

</div>

## 📋 Abstract

We propose **HIEU** (**H**ypernetwork-**I**ntegrated **E**xpert **U**nit), a novel architecture that dynamically generates context-conditioned low-rank weight adaptations based on a rich multi-view context vector fusing regime, time-evolving cross-asset graph relationships, and multi-scale frequency patterns. Our approach enables sample-specific, regime-aware forecasting that captures inter-asset dependencies and temporal shifts while remaining parameter-efficient with intrinsic glass-box explainability.

## ✨ Key Features

- 🎯 **Regime-Aware**: Automatically detects market regimes (Bull/Bear/Volatile/Sideways)
- 📊 **Multi-Asset**: Jointly forecasts 19 cryptocurrency assets with cross-asset dependencies  
- 🔍 **Explainable**: Glass-box interpretability through regime attention and dynamic graphs

## 📈 Results

HIEU achieves **state-of-the-art** performance across all 19 cryptocurrency assets:

| Rank | Model | MAE | RMSE | Interpretable |
|:----:|-------|:---:|:----:|:-------------:|
| 🥇 | **HIEU** | **0.5434±0.0000** | **0.7563±0.0001** | ✅ |
| 🥈 | SimpleMoLE | 0.5445±0.0000 | 0.7571±0.0000 | ❌ |
| 🥉 | iTransformer | 0.5457±0.0001 | 0.7592±0.0002 | ❌ |
| 4 | FEDformer | 0.5467±0.0015 | 0.7606±0.0014 | ❌ |
| 5 | PatchTST | 0.5476±0.0008 | 0.7617±0.0009 | ❌ |
| 6 | Linear | 0.5491±0.0000 | 0.7621±0.0000 | ❌ |
| 7 | DLinear | 0.5493±0.0001 | 0.7623±0.0001 | ❌ |
| 8 | RLinear | 0.5521±0.0000 | 0.7656±0.0000 | ❌ |
| 9 | NLinear | 0.5524±0.0000 | 0.7660±0.0000 | ❌ |
| 10 | Autoformer | 0.5662±0.0152 | 0.7810±0.0155 | ❌ |

<p align="center">
<img src="plots/barplot.png" width="45%">
<img src="plots/boxplot.png" width="45%">
</p>

## 🔬 Visualizations

<p align="center">
<img src="plots/regime.png" width="45%">
<img src="plots/candle.png" width="45%">
</p>

<p align="center">
<img src="plots/lineplot.png" width="45%">
<img src="plots/energy.png" width="45%">
</p>

## 📁 Repository Structure

```
HIEU/
├── data/                   # Cryptocurrency OHLCV data (19 assets)
├── models/
│   └── HIEU/              # HIEU model implementation
│       ├── model.py       # Main model architecture
│       ├── configs.py     # Model configurations
│       └── modules/       # Model components
├── baseline_models/        # Baseline implementations
│   ├── linear_models.py   # Linear, DLinear, NLinear
│   ├── rlinear_model.py   # RLinear
│   ├── patchtst_model.py  # PatchTST
│   ├── itransformer_model.py  # iTransformer
│   ├── Autoformer/        # Autoformer
│   └── FEDformer/         # FEDformer
├── scripts/               # Benchmark scripts
├── analysis/              # Results & analysis
├── plots/                 # Paper figures
└── figures/               # Architecture diagrams
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/nguyenhuyduchieu/HIEU.git
cd HIEU
pip install torch numpy pandas scikit-learn
```

### Run Benchmark

```bash
# Main benchmark (HIEU + 7 baselines)
python scripts/run_benchmark.py

# Individual models
python scripts/run_autoformer_only.py
python scripts/run_fedformer_only.py
```

### Train HIEU

```python
import torch
from models.HIEU.model import HIEUModel, HIEUConfig
from models.HIEU.multi_asset_loader import create_multiasset_loaders

# Load data
train_loader, valid_loader, test_loader, _ = create_multiasset_loaders(
    data_dir='data',
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    seq_len=96, pred_len=96, batch_size=32
)

# Create and train model
config = HIEUConfig()
config.num_nodes = 3
model = HIEUModel(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
for epoch in range(50):
    for x, y in train_loader:
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 📊 Data

Minute-level OHLCV data for **19 cryptocurrencies** from Binance (Oct 2020 - Oct 2025):

| | | | |
|---|---|---|---|
| BTC | ETH | BNB | SOL |
| XRP | ADA | DOT | LINK |
| LTC | BCH | ATOM | XLM |
| ETC | VET | TRX | FIL |
| UNI | DOGE | XMR | |

## 🏗️ Architecture

HIEU consists of four main components:

1. **Regime Encoder** - Detects market regimes via Gumbel-Softmax
2. **Dynamic Graph** - Learns time-evolving cross-asset correlations  
3. **Frequency Bank** - Extracts multi-scale temporal patterns
4. **HyperLinear** - Generates sample-specific prediction weights

## 📝 Citation

```bibtex
@inproceedings{hieu2026,
  title={HIEU: Regime-Aware Hypernetwork Experts for Explainable Multi-Asset Cryptocurrency Forecasting},
  author={Nguyen, Huy Duc Hieu},
  booktitle={IJCAI},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ for IJCAI 2026
</div>
