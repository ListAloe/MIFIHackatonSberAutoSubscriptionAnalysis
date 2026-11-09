"""REST-API: вероятность того, что визит на сайт «СберАвтоподписка» закончится целевым действием.

Запуск из корня репозитория:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import CAT_FEATURES, build_features

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv('MODEL_DIR', BASE_DIR / 'data' / 'models'))
META_PATH = MODEL_DIR / 'catboost_sessions_v2_meta.json'

meta = json.loads(META_PATH.read_text(encoding='utf-8'))
FEATURES: list[str] = meta['features']
CAT_IN_MODEL = [c for c in FEATURES if c in CAT_FEATURES]
THRESHOLD: float = float(meta.get('threshold', 0.5))

model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / meta['model']))

app = FastAPI(
    title='СберАвтоподписка: конверсия визита',
    description='Предсказывает, совершит ли посетитель целевое действие '
                '(заявка, звонок, диалог) по данным визита: UTM, устройство, гео, время.',
    version='2.0.0',
)


class Visit(BaseModel):
    """Данные визита в формате ga_sessions. Все поля, кроме времени, необязательны."""
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    visit_date: str = Field(..., examples=['2021-12-20'])
    visit_time: str = Field(..., examples=['14:05:00'])
    visit_number: int = Field(1, ge=1)
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_screen_resolution: Optional[str] = Field(None, examples=['393x851'])
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None


class PredictRequest(BaseModel):
    data: List[Visit] = Field(..., min_length=1)


class Prediction(BaseModel):
    session_id: Optional[str]
    probability: float
    prediction: int  # 1 — ожидаем целевое действие, 0 — нет


class PredictResponse(BaseModel):
    threshold: float
    predictions: List[Prediction]


@app.get('/health')
def health():
    return {'status': 'ok', 'model': meta['model'], 'roc_auc_test': meta.get('roc_auc', {}).get('test')}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    raw = pd.DataFrame([v.model_dump() for v in request.data])
    try:
        X = build_features(raw)[FEATURES]
    except Exception as e:  # некорректные входные данные
        raise HTTPException(status_code=422, detail=f'Ошибка подготовки признаков: {e}')

    proba = model.predict_proba(X)[:, 1]
    return PredictResponse(
        threshold=THRESHOLD,
        predictions=[
            Prediction(session_id=sid, probability=float(p), prediction=int(p >= THRESHOLD))
            for sid, p in zip(raw['session_id'], proba)
        ],
    )
