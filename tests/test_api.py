import fakeredis
from fastapi.testclient import TestClient

from fraud_engine.features import MemoryVelocityStore
from fraud_engine.serve import RedisVelocityStore, create_app

TX = dict(step=10, type="TRANSFER", amount=5000.0, nameOrig="C1", oldbalanceOrg=5000.0,
          newbalanceOrig=0.0, nameDest="C9", oldbalanceDest=0.0, newbalanceDest=5000.0)


def client(artifacts, tmp_path, store=None):
    app = create_app(store=store or MemoryVelocityStore(), artifact_dir=artifacts,
                     audit_db=str(tmp_path / "a.db"))
    return TestClient(app)


def test_health(artifacts, tmp_path):
    assert client(artifacts, tmp_path).get("/health").json()["status"] == "ok"


def test_score_shape_and_audit(artifacts, tmp_path):
    c = client(artifacts, tmp_path)
    r = c.post("/score", json=TX)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_prob"] <= 1.0
    assert body["action"] in ("ALLOW", "BLOCK")


def test_bad_payload_rejected(artifacts, tmp_path):
    assert client(artifacts, tmp_path).post("/score", json={"step": 1}).status_code == 422


def test_redis_store_matches_memory_store():
    mem, red = MemoryVelocityStore(), RedisVelocityStore(fakeredis.FakeRedis())
    seq = [(1, "A", "B", 10.0), (5, "A", "B", 5.0), (5, "X", "B", 1.0), (40, "A", "B", 1.0)]
    for s in seq:
        assert mem.observe(*s) == red.observe(*s)
