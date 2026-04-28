# Backend Runbook

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure

```bash
cp .env.example .env
```

Recommended env updates:
- `SECRET_KEY` set to strong random value
- `DATABASE_URL` switch to PostgreSQL for production
- `KAGGLE_DATASET_SLUG` set to your preferred heart dataset

## 3) Download Data from Kaggle

```bash
python -m ml.download_kaggle_data --dataset-slug fedesoriano/heart-failure-prediction
```

You can repeat with multiple datasets and keep files in `data/raw`.

## 4) Train Model

```bash
python -m ml.train
```

Outputs:
- `model_artifacts/heart_model.joblib`
- `model_artifacts/metrics.json`
- `data/processed/canonical_training_data.csv`

## 5) Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Optional Makefile Shortcuts

```bash
make install
make download
make train
make run
```

## 6) Auth + Prediction API Quick Test

### Register

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"DemoPass123","full_name":"Demo"}'
```

### Login

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"DemoPass123"}'
```

### Predict (replace TOKEN)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predictions \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "age":57,
    "sex":1,
    "cp":0,
    "trestbps":145,
    "chol":233,
    "fbs":1,
    "restecg":2,
    "thalach":150,
    "exang":0,
    "oldpeak":2.3,
    "slope":0,
    "ca":0,
    "thal":1
  }'
```
