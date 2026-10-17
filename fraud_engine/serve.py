"""FastAPI scoring service.

Velocity state: Redis when REDIS_URL is set, otherwise in-process memory.
Audit log: SQLite file (AUDIT_DB, default audit.db). No external database is
needed, so there is nothing to expire on a free hosting tier.
"""
import json
import os
import sqlite3
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

from .features import FEATURES, WINDOW_STEPS, MemoryVelocityStore, build_vector

ART = os.environ.get("ARTIFACT_DIR", "artifacts")


class RedisVelocityStore:
    """Rolling window in Redis sorted sets (score = step)."""

    def __init__(self, client, window=WINDOW_STEPS):
        self.r = client
        self.window = window

    def observe(self, step, orig, dest, amount):
        lo = step - self.window
        ko, kd = "v:o:" + orig, "v:d:" + dest
        pipe = self.r.pipeline()
        pipe.zremrangebyscore(ko, "-inf", lo)
        pipe.zremrangebyscore(kd, "-inf", lo)
        pipe.zrange(ko, 0, -1)
        pipe.zrange(kd, 0, -1)
        _, _, o_members, d_members = pipe.execute()
        # member format "step|uid|value"
        amt_sum = 0.0
        for m in o_members:
            m = m.decode() if isinstance(m, bytes) else m
            amt_sum += float(m.split("|")[2])
        distinct = len({(m.decode() if isinstance(m, bytes) else m).split("|")[2] for m in d_members})
        uid = "%d" % time.time_ns()
        pipe = self.r.pipeline()
        pipe.zadd(ko, {"%d|%s|%s" % (step, uid, amount): step})
        pipe.zadd(kd, {"%d|%s|%s" % (step, uid, orig): step})
        pipe.expire(ko, 86400)
        pipe.expire(kd, 86400)
        pipe.execute()
        return len(o_members), amt_sum, distinct


def make_store():
    url = os.environ.get("REDIS_URL")
    if url:
        import redis
        return RedisVelocityStore(redis.Redis.from_url(url))
    return MemoryVelocityStore()


class Tx(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


def create_app(store=None, artifact_dir=ART, audit_db=None):
    app = FastAPI(title="Fraud Detection Engine")
    sess = ort.InferenceSession(os.path.join(artifact_dir, "model.onnx"),
                                providers=["CPUExecutionProvider"])
    cfg = json.load(open(os.path.join(artifact_dir, "threshold.json")))
    assert cfg["features"] == FEATURES, "feature list mismatch between model and code"
    store = store or make_store()
    db = sqlite3.connect(audit_db or os.environ.get("AUDIT_DB", "audit.db"), check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS audit (ts REAL, name_orig TEXT, step INT, amount REAL,"
               " fraud_prob REAL, action TEXT, latency_ms REAL)")

    @app.get("/health")
    def health():
        return {"status": "ok", "threshold": cfg["threshold"]}

    @app.post("/score")
    def score(tx: Tx):
        t0 = time.perf_counter()
        d = tx.model_dump()
        v = store.observe(d["step"], d["nameOrig"], d["nameDest"], d["amount"])
        x = np.asarray([build_vector(d, v)], dtype=np.float32)
        out = sess.run(None, {"input": x})
        p = out[1]
        prob = float(p[0][1]) if not isinstance(p[0], dict) else float(p[0][1])
        action = "BLOCK" if prob >= cfg["threshold"] else "ALLOW"
        ms = (time.perf_counter() - t0) * 1000
        db.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?)",
                   (time.time(), d["nameOrig"], d["step"], d["amount"], prob, action, ms))
        db.commit()
        return {"fraud_prob": round(prob, 4), "action": action, "latency_ms": round(ms, 3)}

    return app


def app_factory():
    return create_app()
