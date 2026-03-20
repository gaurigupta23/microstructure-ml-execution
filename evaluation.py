import numpy as np


def simulate_execution(df, schedule, side="buy", impact_coeff=0.02, impact_exp=0.5):
    side = side.lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    mids = df["mid"].to_numpy()
    spreads = df["spread"].to_numpy()
    volumes = df["volume"].to_numpy() + 1e-6
    trades = np.asarray(schedule, dtype=float)

    if len(trades) > len(mids):
        raise ValueError("schedule longer than data length")

    arrival_price = mids[0]
    costs = []
    exec_prices = []

    for i, qty in enumerate(trades):
        if qty <= 0:
            continue
        mid = mids[i]
        spread = spreads[i]
        vol = volumes[i]
        impact = impact_coeff * (qty / vol) ** impact_exp
        if side == "buy":
            price = mid + spread / 2.0 + impact
        else:
            price = mid - spread / 2.0 - impact
        exec_prices.append(price)
        costs.append(qty * price)

    total_qty = trades.sum()
    total_cost = float(np.sum(costs))
    avg_price = total_cost / max(total_qty, 1e-9)
    if side == "buy":
        shortfall = total_cost - total_qty * arrival_price
    else:
        shortfall = total_qty * arrival_price - total_cost

    slippage_bps = 1e4 * shortfall / max(total_qty * arrival_price, 1e-9)

    return {
        "total_qty": float(total_qty),
        "total_cost": total_cost,
        "avg_price": avg_price,
        "arrival_price": arrival_price,
        "shortfall": shortfall,
        "slippage_bps": slippage_bps,
        "exec_prices": exec_prices,
    }
