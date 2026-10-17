"""Offline (training replay) and online (API) feature values must match."""
import numpy as np

from fraud_engine.data import generate
from fraud_engine.features import MemoryVelocityStore, build_matrix, build_vector


def test_offline_matches_online():
    df = generate(2000, seed=3).sort_values("step", kind="stable")
    offline = build_matrix(df)
    store = MemoryVelocityStore()
    online = []
    for r in df.itertuples(index=False):
        t = r._asdict()
        online.append(build_vector(t, store.observe(t["step"], t["nameOrig"], t["nameDest"], t["amount"])))
    assert np.allclose(offline, np.asarray(online, dtype=np.float32))
