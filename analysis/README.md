# Analysis Results

This folder contains benchmark results and interpretability analysis for HIEU.

## Files

- `complete_benchmark_results.csv` - Full benchmark results (all models × all assets × 3 seeds)
- `ablation_study.csv` - Ablation study results
- `ablation_summary.txt` - Ablation study summary
- `regime_analysis.csv` - Market regime analysis (4 regimes discovered)
- `frequency_importance.csv` - Frequency band importance weights
- `graph_adjacency.csv` - Learned cross-asset correlation matrix

## Benchmark Results Summary

| Rank | Model | MAE | RMSE |
|------|-------|-----|------|
| 1 | HIEU | 0.5434±0.0000 | 0.7563±0.0001 |
| 2 | SimpleMoLE | 0.5445±0.0000 | 0.7571±0.0000 |
| 3 | iTransformer | 0.5457±0.0001 | 0.7592±0.0002 |
| 4 | FEDformer | 0.5467±0.0015 | 0.7606±0.0014 |
| 5 | PatchTST | 0.5476±0.0008 | 0.7617±0.0009 |
| 6 | Linear | 0.5491±0.0000 | 0.7621±0.0000 |
| 7 | Autoformer | 0.5662±0.0152 | 0.7810±0.0155 |

## Regime Analysis

HIEU discovers 4 latent market regimes:
- Regime 0: Bearish/Corrective (negative returns, high volatility)
- Regime 1: Transitional
- Regime 2: Sideways/Consolidation
- Regime 3: Bullish (positive returns, low volatility)
