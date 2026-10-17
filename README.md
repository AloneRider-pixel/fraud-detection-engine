# Fraud Detection Engine

A real-time transaction scoring service: FastAPI, rolling-window velocity features
(in-memory or Redis), an XGBoost model served through ONNX Runtime, and an audit log.

```
POST /score  ->  velocity features (Redis or memory)  ->  ONNX model  ->  {"fraud_prob", "action"}
                                                          |
                                                          +--> audit log (SQLite)
```

## What is real and what is not

- The data is **synthetic**, generated in `fraud_engine/data.py` with PaySim column names.
  Real PaySim needs a Kaggle login. To use it, download the CSV and run
  `python -m fraud_engine.train --csv PaySim.csv`.
- Metrics on synthetic data only show that the pipeline works. They are not a claim about real fraud.
- Audit log is SQLite, not PostgreSQL. This avoids a managed database that expires on free tiers.

## Design decisions

- **Time-based split.** Train on earlier steps, validate and test on later ones. No random shuffle.
- **Imbalance.** `scale_pos_weight` in XGBoost, judged by PR-AUC (not accuracy).
  If you add SMOTE, apply it only inside training folds (for example `imblearn.pipeline.Pipeline`), never before the split.
- **Threshold.** Chosen on the validation split only: highest recall with precision at or above a target
  (`--min-precision`, default 0.90). Test is touched once for reporting.
- **No train/serve skew.** Training replays rows through the same velocity store class used by the API.
  `tests/test_parity.py` checks offline and online features match, and `tests/test_api.py` checks the Redis store
  returns the same values as the in-memory store.
- **ONNX.** The model is exported with onnxmltools and served with ONNX Runtime. Speed is measured, not assumed (below).

## Run it

```bash
pip install -r requirements-dev.txt
python -m fraud_engine.train          # writes artifacts/model.onnx and artifacts/metrics.json
python -m pytest -q
uvicorn fraud_engine.serve:app_factory --factory --port 8000
curl -s localhost:8000/score -H 'content-type: application/json' -d '{"step":10,"type":"TRANSFER","amount":5000,"nameOrig":"C1","oldbalanceOrg":5000,"newbalanceOrig":0,"nameDest":"C9","oldbalanceDest":0,"newbalanceDest":5000}'
```

With Redis for velocity state: `docker compose up --build` (API on port 8000, Redis as a service).

## Measured results (synthetic data, seed 0, 60,000 rows, 1% fraud)

Produced by `python -m fraud_engine.train` in a Python 3.10 sandbox. Run it yourself to reproduce.

| Metric | Value |
|---|---|
| Test PR-AUC | 0.44 |
| Threshold (from validation, precision target 0.90) | 0.95 |
| Test precision / recall at that threshold | 0.50 / 0.43 (117 fraud rows in test) |
| Scoring latency, in-process, `scripts/bench.py` (1000 calls, includes test-client overhead) | p50 5.2 ms, p95 6.1 ms, p99 8.7 ms |
| Server-side model + features time reported by the API for a single call | about 0.5 ms |

The synthetic generator adds noise on purpose, so these numbers are modest. The precision target of 0.90 on validation
was not met on test, which is the kind of gap you get from a small validation set with 1% fraud. Latency was measured
on a sandbox machine, not production hardware, and not over HTTP/network.
Docker image build was not run in the sandbox that produced this repo, CI builds it on every push.

## Deploy

Free-tier option: Render web service from the `Dockerfile` (`render.yaml` included). The free plan sleeps after
inactivity, so the first request after idle is slow. Without `REDIS_URL` the API uses in-memory velocity state and
SQLite audit storage, which reset on restart. That is fine for a demo, not for production.

## Limits

- Velocity state is per API process unless Redis is configured.
- Drift monitoring is not implemented yet. Next step: log score distribution and feature means from the audit table and compare to training (PSI).
- No authentication on `/score`.
