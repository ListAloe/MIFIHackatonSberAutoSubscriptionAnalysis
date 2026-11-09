"""Признаки визита для модели конверсии «СберАвтоподписка».

Один и тот же код используется и при обучении (notebooks/03_model.ipynb),
и в API (src/api.py), чтобы признаки в проде совпадали с признаками при обучении.

Используются только данные, известные в момент начала визита: UTM-метки, устройство,
гео, дата/время визита и порядковый номер визита. Ничего из хитов (что человек делал
на сайте) в признаки не попадает — иначе модель «подсматривает» ответ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Целевые действия (ga_hits.event_action): заявка, звонок, диалог и т.п.
TARGET_ACTIONS = [
    'sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
    'sub_custom_question_submit_click', 'sub_call_number_click',
    'sub_callback_submit_click', 'sub_submit_success', 'sub_car_request_submit_click',
]

# Органический трафик по глоссарию задачи
ORGANIC_MEDIUMS = ('organic', 'referral', '(none)')

MISSING = '(missing)'

# Сырые поля визита, которые нужны для построения признаков
RAW_COLUMNS = [
    'visit_date', 'visit_time', 'visit_number',
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
    'device_category', 'device_os', 'device_brand', 'device_screen_resolution',
    'device_browser', 'geo_country', 'geo_city',
]

CAT_FEATURES = [
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
    'device_category', 'device_os', 'device_brand', 'device_browser',
    'geo_country', 'geo_city',
    'source_medium',          # связка «источник + тип трафика»
    'device_category_os',     # связка «тип устройства + ОС»
]

NUM_FEATURES = [
    'visit_number', 'is_new_visitor', 'visit_hour', 'visit_weekday', 'is_weekend',
    'is_organic', 'screen_w', 'screen_h', 'screen_ratio',
]


def _clean_cat(s: pd.Series) -> pd.Series:
    s = s.astype('string').str.strip()
    s = s.replace({'': pd.NA, '(not set)': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    return s.fillna(MISSING).astype(str)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Строит признаки из сырых полей визита (формат ga_sessions).

    Возвращает DataFrame со всеми колонками CAT_FEATURES + NUM_FEATURES.
    Отсутствующие сырые поля считаются пропусками.
    """
    df = raw.copy()
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    out = pd.DataFrame(index=df.index)

    base_cats = [c for c in CAT_FEATURES if c not in ('source_medium', 'device_category_os')]
    for col in base_cats:
        out[col] = _clean_cat(df[col])
    out['source_medium'] = out['utm_source'] + ' | ' + out['utm_medium']
    out['device_category_os'] = out['device_category'] + ' | ' + out['device_os']

    # Время визита
    # visit_date может прийти строкой или datetime, visit_time — строкой 'ЧЧ:ММ:СС' или time
    date = pd.to_datetime(df['visit_date'], errors='coerce').dt.normalize()
    time = pd.to_timedelta(df['visit_time'].astype(str), errors='coerce')
    dt = date + time
    out['visit_hour'] = dt.dt.hour.fillna(-1).astype('int16')
    out['visit_weekday'] = dt.dt.weekday.fillna(-1).astype('int16')
    out['is_weekend'] = (out['visit_weekday'] >= 5).astype('int8')

    # Номер визита: новый посетитель или вернувшийся
    vn = pd.to_numeric(df['visit_number'], errors='coerce').fillna(1)
    out['visit_number'] = vn.clip(upper=50).astype('int32')
    out['is_new_visitor'] = (vn <= 1).astype('int8')

    # Органика vs реклама
    out['is_organic'] = df['utm_medium'].isin(ORGANIC_MEDIUMS).astype('int8')

    # Экран
    res = (df['device_screen_resolution'].astype('string')
           .str.extract(r'(\d+)\s*x\s*(\d+)').astype('float32'))
    out['screen_w'] = res[0]
    out['screen_h'] = res[1]
    out['screen_ratio'] = (out['screen_h'] / out['screen_w']).replace([np.inf, -np.inf], np.nan)

    return out[CAT_FEATURES + NUM_FEATURES]


def session_targets(hits: pd.DataFrame) -> pd.Series:
    """session_id → 1, если в сессии было хотя бы одно целевое действие."""
    flag = hits['event_action'].isin(TARGET_ACTIONS)
    return flag.groupby(hits['session_id']).any().astype('int8')
