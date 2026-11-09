"""Тесты API. Запуск из корня репозитория: pytest -q"""
import time

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VISIT = {
    'session_id': '5692861315757623740.1632356796.1632356796',
    'client_id': '2108382700.1637753791',
    'visit_date': '2021-12-20',
    'visit_time': '14:05:00',
    'visit_number': 1,
    'utm_source': 'ZpYIoDJMcFzVoPFsHGJL',
    'utm_medium': 'banner',
    'utm_campaign': 'LEoPHuyFvzoNfnzGgfcd',
    'utm_adcontent': 'vCIpmpaGBnIQhyYNkXqp',
    'utm_keyword': 'puhZPIYqKXeFPaUviSjo',
    'device_category': 'mobile',
    'device_os': 'Android',
    'device_brand': 'Huawei',
    'device_screen_resolution': '360x720',
    'device_browser': 'Chrome',
    'geo_country': 'Russia',
    'geo_city': 'Moscow',
}


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_predict_one():
    r = client.post('/predict', json={'data': [VISIT]})
    assert r.status_code == 200
    body = r.json()
    p = body['predictions'][0]
    assert p['session_id'] == VISIT['session_id']
    assert 0.0 <= p['probability'] <= 1.0
    assert p['prediction'] in (0, 1)
    assert p['prediction'] == int(p['probability'] >= body['threshold'])


def test_predict_minimal_fields():
    """Только обязательные поля — остальное считается пропуском."""
    r = client.post('/predict', json={'data': [{'visit_date': '2021-12-20', 'visit_time': '09:00:00'}]})
    assert r.status_code == 200


def test_organic_vs_banner_differ():
    organic = {**VISIT, 'utm_medium': 'organic', 'utm_source': 'fDLlAcSmythWSCVMvqvL'}
    r = client.post('/predict', json={'data': [VISIT, organic]})
    probs = [p['probability'] for p in r.json()['predictions']]
    assert probs[0] != probs[1]


def test_bad_request():
    r = client.post('/predict', json={'data': [{'visit_number': 'abc'}]})
    assert r.status_code == 422


def test_response_time_1000_visits():
    """Требование брифа: ответ ≤ 3 секунд."""
    t = time.perf_counter()
    r = client.post('/predict', json={'data': [VISIT] * 1000})
    assert r.status_code == 200
    assert time.perf_counter() - t < 3.0
