"""Synthetic PaySim-style transaction generator.

The real PaySim dataset (Kaggle) needs a login to download. This generator
produces data with the same column names so the pipeline runs end to end with
no download. Results on synthetic data say nothing about real-world accuracy.
To use real PaySim, pass its CSV path to `train.py --csv`.
"""
import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]


def generate(n=60_000, fraud_rate=0.01, seed=0, n_accounts=8_000):
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, 721, n))
    ttype = rng.choice(TYPES, n, p=[0.34, 0.22, 0.35, 0.08, 0.01])
    orig = rng.integers(0, n_accounts, n)
    dest = rng.integers(0, n_accounts, n)
    old_org = np.round(rng.lognormal(8.5, 1.6, n), 2)
    amount = np.round(np.minimum(rng.lognormal(6.5, 1.4, n), old_org), 2)
    is_fraud = np.zeros(n, dtype=int)

    eligible = np.where(np.isin(ttype, ["TRANSFER", "CASH_OUT"]))[0]
    n_fraud = int(n * fraud_rate)
    idx = rng.choice(eligible, n_fraud, replace=False)
    is_fraud[idx] = 1
    # Fraud pattern: drain most of the account, often a burst from one origin
    # account and a small set of mule destination accounts. Some noise so the
    # classes overlap.
    drain = rng.random(n_fraud) < 0.8
    amount[idx] = np.where(drain, old_org[idx], amount[idx] * 4)
    amount[idx] = np.round(np.minimum(amount[idx], old_org[idx] * 1.0), 2)
    mules = rng.integers(0, 40, n_fraud) + n_accounts
    dest[idx] = mules
    burst = rng.random(n_fraud) < 0.5
    fraud_origs = rng.integers(0, 200, n_fraud) + 2 * n_accounts
    orig[idx[burst]] = fraud_origs[burst]

    new_org = np.round(np.maximum(old_org - amount, 0), 2)
    old_dst = np.round(rng.lognormal(8.0, 1.8, n), 2)
    new_dst = np.round(old_dst + amount, 2)
    # Legit rows sometimes have inconsistent balances (noise)
    noisy = (rng.random(n) < 0.05) & (is_fraud == 0)
    new_org[noisy] = old_org[noisy]

    return pd.DataFrame({
        "step": step, "type": ttype, "amount": amount,
        "nameOrig": ["C%d" % i for i in orig],
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "nameDest": ["C%d" % i for i in dest],
        "oldbalanceDest": old_dst, "newbalanceDest": new_dst,
        "isFraud": is_fraud,
    })
