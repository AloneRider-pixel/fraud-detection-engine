"""Measure in-process scoring latency. Run after training; prints your numbers."""
import statistics
import tempfile
import time

from fraud_engine.features import MemoryVelocityStore
from fraud_engine.serve import create_app
from fastapi.testclient import TestClient

TX = dict(step=10, type="TRANSFER", amount=5000.0, nameOrig="C1", oldbalanceOrg=5000.0,
          newbalanceOrig=0.0, nameDest="C9", oldbalanceDest=0.0, newbalanceDest=5000.0)

app = create_app(store=MemoryVelocityStore(), audit_db=tempfile.mktemp())
c = TestClient(app)
for _ in range(50):
    c.post("/score", json=TX)
lat = []
for i in range(1000):
    t = dict(TX, step=10 + i % 5, nameOrig="C%d" % (i % 100))
    t0 = time.perf_counter()
    c.post("/score", json=t)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
print("n=1000 p50=%.2fms p95=%.2fms p99=%.2fms mean=%.2fms" % (
    lat[500], lat[950], lat[990], statistics.mean(lat)))
