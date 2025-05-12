from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier, Pool

app = FastAPI(title="SberAuto Subscription Conversion API",
              description="Predict probability of target conversion per hit record",
              version="1.0.0")

# Пути к данным и модели
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "data" / "model_data" / "catboost_optuna_m1.cbm"
PROCESSED_PATH = BASE_DIR / "data" / "processed_data" / "data_processed.pkl"

# Загружаем модель
try:
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

# Если у вас есть готовый препроцессор (например, сохраненный в pickle), можно загрузить его:
# from joblib import load
# preprocessor = load(PROCESSED_PATH)

# Но в нашем случае предположим, что препроцессинг минимальный
# и мы просто подаем табличные данные сразу в модель.

# Список категориальных признаков, как при обучении
CAT_FEATURES = [
    'visit_hour', 'visit_weekday', 'is_weekend', 'visit_number',
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
    'device_category', 'device_os', 'device_brand', 'device_browser',
    'geo_country', 'geo_city'
]

class Record(BaseModel):
    """Один объект запроса — словарь с признаками."""
    session_id: str
    client_id: str
    visit_hour: int
    visit_weekday: int
    is_weekend: int
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_screen_height: int
    device_screen_width: int
    device_browser: str
    geo_country: str
    geo_city: str
    hit_number: int
    aspect_ratio: float
    n_hits: int
    n_target_hits: int
    n_unique_pages: int
    visit_day: int
    visit_month: int
    visit_year: int
    hit_sec: float
    session_total_sec: float

class PredictRequest(BaseModel):
    data: List[Record] = Field(..., description="Список записей для предсказания")

class PredictResponse(BaseModel):
    session_id: List[str]
    client_id: List[str]
    probability: List[float]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Преобразуем вход в DataFrame
    try:
        df = pd.DataFrame([r.dict() for r in request.data])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка чтения данных: {e}")

    # Проверим, что все нужные колонки есть
    missing = set(CAT_FEATURES + ['hit_number','aspect_ratio','n_hits',
                                  'n_target_hits','n_unique_pages',
                                  'visit_day','visit_month','visit_year',
                                  'hit_sec','session_total_sec']) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Входные данные не содержат обязательные признаки: {missing}"
        )

    # Выделяем признаки для модели
    X = df.drop(['session_id', 'client_id'], axis=1)

    # Создаем Pool для корректной обработки категорий
    pool = Pool(data=X, cat_features=CAT_FEATURES)

    # Предсказания
    try:
        preds = model.predict_proba(pool)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")

    return PredictResponse(
        session_id=df['session_id'].tolist(),
        client_id=df['client_id'].tolist(),
        probability=preds.tolist()
    )

@app.get("/health")
def health():
    return {"status": "ok"}
