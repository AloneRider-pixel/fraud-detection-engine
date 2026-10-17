from fraud_engine.data import generate
from fraud_engine.features import FEATURES, MemoryVelocityStore, build_matrix, build_vector


def tx(**kw):
    base = dict(step=10, type="TRANSFER", amount=100.0, nameOrig="C1", oldbalanceOrg=100.0,
                newbalanceOrig=0.0, nameDest="C2", oldbalanceDest=50.0, newbalanceDest=150.0)
    base.update(kw)
    return base


def test_vector_length_matches_feature_list():
    assert len(build_vector(tx(), (0, 0.0, 0))) == len(FEATURES)


def test_drained_flag_and_balance_error():
    v = dict(zip(FEATURES, build_vector(tx(), (0, 0.0, 0))))
    assert v["orig_drained"] == 1.0
    assert v["orig_balance_error"] == 0.0
    assert v["type_TRANSFER"] == 1.0


def test_velocity_counts_previous_only_and_expires():
    s = MemoryVelocityStore(window=24)
    assert s.observe(1, "A", "B", 10.0) == (0, 0.0, 0)      # no self-leak
    assert s.observe(5, "A", "B", 5.0) == (1, 10.0, 1)
    assert s.observe(5, "X", "B", 1.0)[2] == 1               # distinct origins to B
    assert s.observe(40, "A", "B", 1.0)[0] == 0              # window expired


def test_generator_has_requested_fraud_rate():
    df = generate(5000, fraud_rate=0.02, seed=1)
    assert abs(df.isFraud.mean() - 0.02) < 0.002
    assert set(df.columns) >= {"step", "type", "amount", "nameOrig", "isFraud"}
