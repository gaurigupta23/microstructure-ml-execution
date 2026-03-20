# Market Microstructure Simulator + ML Execution

A self-contained Python project that simulates a stylised limit order book (LOB), trains a short-horizon ML price-direction model, and benchmarks four execution strategies — **TWAP, VWAP, Almgren–Chriss, and ML-adaptive** — on realistic synthetic market data.

Built as a portfolio piece for quant/trading interviews. The code is intentionally transparent: every module is a single file with no hidden dependencies.

---

## What this demonstrates

| Area | Detail |
|---|---|
| **Market microstructure** | Order-flow imbalance, OFI, spread dynamics, market impact |
| **LOB simulation** | Poisson arrivals/cancels, depth-weighted mid-price evolution |
| **ML for finance** | Feature engineering from book data, time-based train/val/test split, logistic regression baseline |
| **Execution research** | Implementation shortfall, slippage (bps), schedule comparison |

---

## Project layout

```
.
├── simulator.py          # Stylised LOB simulator (LOBSimulator class)
├── ml.py                 # Feature engineering, model training & evaluation
├── strategies.py         # TWAP / VWAP / Almgren-Chriss / ML-adaptive schedules
├── evaluation.py         # Execution cost metrics (shortfall, slippage bps)
├── microstructure_ml_execution.ipynb  # Full narrative walkthrough with charts
├── environment.yml       # Conda environment (recommended)
└── requirements.txt      # pip fallback
```

---

## Quick start

### Option 1 — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate mm_sim
python -m ipykernel install --user --name mm_sim --display-name "mm_sim"
jupyter notebook microstructure_ml_execution.ipynb
```

Select the **mm_sim** kernel when the notebook opens.

### Option 2 — pip

```bash
python -m pip install -r requirements.txt
jupyter notebook microstructure_ml_execution.ipynb
```

---

## Module overview

### `simulator.py` — `LOBSimulator`

Simulates a stylised two-sided order book at each tick:

- Limit order **adds** and **cancels** arrive as Poisson processes
- Market orders consume depth; net order-flow imbalance and OFI bias the next-tick price move via a sigmoid
- Spread widens with market-order activity
- Returns a `pd.DataFrame` with `mid`, `spread`, `imbalance`, `ofi`, `volume`, `return`, and per-level bid/ask depths

```python
from simulator import LOBSimulator
sim = LOBSimulator(seed=42, k_imbalance=3.5, k_ofi=0.12)
df = sim.simulate(num_steps=3000)
```

### `ml.py` — short-horizon signal

Predicts **next-tick direction** (up vs. down) from book features using a logistic regression pipeline with standard scaling.

- `make_features(df, horizon=1)` — builds feature matrix and binary target
- `time_split(X, y)` — strict chronological 70/15/15 split (no lookahead)
- `train_model` / `evaluate_model` — scikit-learn Pipeline; returns accuracy and AUC

### `strategies.py` — execution schedules

| Function | Strategy |
|---|---|
| `twap_schedule` | Equal slice every step |
| `vwap_schedule` | Slice proportional to observed volume |
| `ac_schedule` | Almgren–Chriss optimal trajectory (`kappa` controls urgency) |
| `ml_adaptive_schedule` | Shifts weight toward high-signal steps using model probabilities |

### `evaluation.py` — `simulate_execution`

Prices each slice at `mid ± half-spread ± market-impact` (power-law model) and computes:

- **Average execution price**
- **Implementation shortfall** (vs. arrival mid)
- **Slippage in basis points**

---

## Results (typical run)

| Strategy | Slippage (bps) |
|---|---|
| TWAP | baseline |
| VWAP | ≈ TWAP ± noise |
| Almgren–Chriss | lower (front-loads risk) |
| ML-Adaptive | lowest when signal is live |

Actual numbers vary with the random seed; run the notebook to reproduce.

---

## Design choices & interview talking points

- **Why logistic regression?** Interpretable, fast, hard to overfit on 3 k ticks. A natural conversation starter about when to add complexity.
- **Time-based split** — critical in finance; random splits leak future information into training.
- **Stylised simulator** — captures the mechanisms that matter (imbalance drives price, impact grows with slice size) without pretending to be a real exchange.
- **Implementation shortfall vs. backtest PnL** — execution cost is not the same as alpha decay; the notebook walks through the difference.
- **Market impact model** — power-law `impact = coeff × (qty/vol)^exp` is standard in academic literature; the notebook discusses sensitivity to the exponent.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical core |
| `pandas` | Data wrangling |
| `scikit-learn` | ML pipeline |
| `matplotlib` / `seaborn` | Visualisation |

Python 3.10+. No GPU required.

---

## License

MIT — free to use, adapt, and extend.
