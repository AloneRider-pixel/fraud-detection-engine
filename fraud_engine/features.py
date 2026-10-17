"""Feature logic shared by training (offline) and serving (online).

Velocity features come from a store with an `observe` method. Training replays
transactions in time order through the same store class the API uses, so the
offline and online values cannot drift apart. tests/test_parity.py checks it.
"""
import math
from collections import defaultdict, deque

import numpy as np

TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]
WINDOW_STEPS = 24  # one PaySim step is one hour
FEATURES = (
    ["amount", "log_amount", "orig_balance_error", "dest_balance_error",
     "orig_drained", "hour", "orig_count_w", "orig_amount_sum_w", "dest_distinct_orig_w"]
    + ["type_" + t for t in TYPES]
)


class MemoryVelocityStore:
    """In-process rolling window. Used for training replay, tests and the
    default local server."""

    def __init__(self, window=WINDOW_STEPS):
        self.window = window
        self.by_orig = defaultdict(deque)   # (step, amount)
        self.by_dest = defaultdict(deque)   # (step, orig)

    def observe(self, step, orig, dest, amount):
        lo = step - self.window
        oq = self.by_orig[orig]
        while oq and oq[0][0] <= lo:
            oq.popleft()
        dq = self.by_dest[dest]
        while dq and dq[0][0] <= lo:
            dq.popleft()
        # Read current window BEFORE adding this transaction (no self-leak)
        count = len(oq)
        amt_sum = sum(a for _, a in oq)
        distinct = len({o for _, o in dq})
        oq.append((step, amount))
        dq.append((step, orig))
        return count, amt_sum, distinct


def build_vector(tx, velocity):
    """tx: dict with step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,
    nameDest,oldbalanceDest,newbalanceDest. velocity: (count, sum, distinct)."""
    amount = float(tx["amount"])
    orig_err = tx["oldbalanceOrg"] - amount - tx["newbalanceOrig"]
    dest_err = tx["oldbalanceDest"] + amount - tx["newbalanceDest"]
    drained = 1.0 if (tx["oldbalanceOrg"] > 0 and tx["newbalanceOrig"] == 0) else 0.0
    row = [amount, math.log1p(amount), orig_err, dest_err, drained,
           float(tx["step"] % 24), float(velocity[0]), float(velocity[1]), float(velocity[2])]
    row += [1.0 if tx["type"] == t else 0.0 for t in TYPES]
    return row


def build_matrix(df, store=None):
    """Replay a time-ordered DataFrame through a velocity store."""
    store = store or MemoryVelocityStore()
    rows = []
    for r in df.sort_values("step", kind="stable").itertuples(index=False):
        tx = r._asdict()
        v = store.observe(tx["step"], tx["nameOrig"], tx["nameDest"], tx["amount"])
        rows.append(build_vector(tx, v))
    return np.asarray(rows, dtype=np.float32)
