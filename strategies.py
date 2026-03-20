import numpy as np


def twap_schedule(total_qty, num_steps):
    qty = float(total_qty)
    return np.full(num_steps, qty / num_steps)


def vwap_schedule(total_qty, volumes):
    qty = float(total_qty)
    vols = np.asarray(volumes, dtype=float)
    total_vol = vols.sum()
    if total_vol <= 0:
        return twap_schedule(qty, len(vols))
    return qty * vols / total_vol


def ac_schedule(total_qty, num_steps, kappa=0.05):
    qty = float(total_qty)
    if kappa <= 0:
        return twap_schedule(qty, num_steps)
    t = np.arange(num_steps + 1, dtype=float)
    x = np.sinh(kappa * (num_steps - t)) / np.sinh(kappa * num_steps)
    x = x * qty
    trades = x[:-1] - x[1:]
    return trades


def ml_adaptive_schedule(total_qty, base_schedule, probs, aggressiveness=0.4):
    qty = float(total_qty)
    base = np.asarray(base_schedule, dtype=float)
    p = np.asarray(probs, dtype=float)
    weights = 1.0 + aggressiveness * (p - 0.5) * 2.0
    weights = np.clip(weights, 0.5, 1.5)
    scaled = base * weights
    scaled_sum = scaled.sum()
    if scaled_sum <= 0:
        return twap_schedule(qty, len(base))
    return qty * scaled / scaled_sum
