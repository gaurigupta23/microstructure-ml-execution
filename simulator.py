import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LOBSimulator:
    """Stylized limit order book simulator for educational use."""

    def __init__(
        self,
        num_levels=3,
        seed=42,
        start_price=100.0,
        tick=0.01,
        base_spread=0.02,
        base_depth=200.0,
        lambda_add=6.0,
        lambda_cancel=4.0,
        lambda_mkt=3.0,
        k_imbalance=2.0,
        k_ofi=0.05,
    ):
        self.num_levels = num_levels
        self.rng = np.random.default_rng(seed)
        self.mid = float(start_price)
        self.tick = float(tick)
        self.base_spread = float(base_spread)
        self.spread = float(base_spread)
        self.base_depth = float(base_depth)
        self.lambda_add = float(lambda_add)
        self.lambda_cancel = float(lambda_cancel)
        self.lambda_mkt = float(lambda_mkt)
        self.k_imbalance = float(k_imbalance)
        self.k_ofi = float(k_ofi)

        self.bid_depths = self._init_depths()
        self.ask_depths = self._init_depths()

    def _init_depths(self):
        noise = self.rng.normal(0.0, 0.15, size=self.num_levels)
        depths = self.base_depth * (1.0 + noise)
        return np.clip(depths, self.base_depth * 0.3, self.base_depth * 3.0)

    def _random_size(self, scale=40.0):
        return float(self.rng.exponential(scale))

    def _apply_adds(self, side_depths, n_add):
        for _ in range(n_add):
            level = int(self.rng.integers(0, self.num_levels))
            side_depths[level] += self._random_size()

    def _apply_cancels(self, side_depths, n_cancel):
        for _ in range(n_cancel):
            level = int(self.rng.integers(0, self.num_levels))
            side_depths[level] = max(0.0, side_depths[level] - self._random_size())

    def step(self):
        prev_mid = self.mid

        add_buy = int(self.rng.poisson(self.lambda_add))
        add_sell = int(self.rng.poisson(self.lambda_add))
        cancel_buy = int(self.rng.poisson(self.lambda_cancel))
        cancel_sell = int(self.rng.poisson(self.lambda_cancel))
        mkt_buy = int(self.rng.poisson(self.lambda_mkt))
        mkt_sell = int(self.rng.poisson(self.lambda_mkt))

        self._apply_adds(self.bid_depths, add_buy)
        self._apply_adds(self.ask_depths, add_sell)
        self._apply_cancels(self.bid_depths, cancel_buy)
        self._apply_cancels(self.ask_depths, cancel_sell)

        bid_total = float(self.bid_depths.sum())
        ask_total = float(self.ask_depths.sum())
        depth_total = bid_total + ask_total
        imbalance = (bid_total - ask_total) / (depth_total + 1e-9)
        ofi = (add_buy - add_sell) + (mkt_buy - mkt_sell) - (cancel_buy - cancel_sell)

        prob_up = _sigmoid(self.k_imbalance * imbalance + self.k_ofi * ofi)
        u = float(self.rng.random())
        if u < prob_up:
            self.mid += self.tick
        elif u > 1.0 - prob_up:
            self.mid -= self.tick

        vol = mkt_buy + mkt_sell
        self.spread = self.base_spread * (1.0 + 0.1 * np.tanh(vol / 5.0))

        mid_return = (self.mid - prev_mid) / max(prev_mid, 1e-9)

        row = {
            "mid": self.mid,
            "spread": self.spread,
            "imbalance": imbalance,
            "ofi": ofi,
            "volume": vol,
            "return": mid_return,
        }
        for i in range(self.num_levels):
            row[f"bid_depth_{i+1}"] = float(self.bid_depths[i])
            row[f"ask_depth_{i+1}"] = float(self.ask_depths[i])
        return row

    def simulate(self, num_steps=2000):
        records = []
        for _ in range(num_steps):
            records.append(self.step())
        df = pd.DataFrame(records)
        df.index.name = "t"
        return df
