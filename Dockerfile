FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fraud_engine fraud_engine
# Train at build time on synthetic data so the image is self-contained.
RUN python -m fraud_engine.train --out artifacts
EXPOSE 8000
CMD ["sh", "-c", "uvicorn fraud_engine.serve:app_factory --factory --host 0.0.0.0 --port ${PORT:-8000}"]
