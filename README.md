# Анализ эффективности сайта «СберАвтоподписка»

Проект направлен на предоставление REST-API для онлайн-предсказания вероятности конверсии пользователя (оставление заявки, заказ звонка и др.) на основании признаков сессии и хитов Google Analytics.

---
MIFIHackatonSberAutoSubscriptionAnalysis/
## 📁 Структура репозитория
```bash

MIFIHackatonSberAutoSubscriptionAnalysis/
├── data
│   ├── model_data
│   │   ├── catboost_optuna_m1.cbm    # Скомпилированная бинарная модель Catboost
│   │   └── catboost_optuna_m1.json   # Метаданные модели
├── LICENSE                           # Лицензия проекта (MIT)
├── notebooks                         # Jupyter-ноутбуки для различных этапов анализа
│   ├── data_preprocessing
│   │   └── data_preprocessing.ipynb  # Ноутбук: чтение raw_data → очистка → сохранение processed_data.pkl
│   ├── EDA
│   │   └── eda.ipynb                 # Ноутбук: визуализации и статистический анализ признаков
│   └── modeling
│       ├── catboost_info/            # Подпапка с вспомогательными артефактами (графики, логи Optuna)
│       ├── catboost.ipynb            # Эксперименты с CatBoost + Optuna-тюнинг гиперпараметров
│       └── model_data_json.ipynb     # Черновой блокнот для работы с JSON метаданными модели
├── README.md                         # Обзор проекта: цели, структура, быстрый старт
├── requirements.txt                  # Список всех pip-зависимостей для воспроизводимости окружения
├── src                               # Исходный код приложения и вспомогательные модули
│   ├── app
│   │   └── app.py                    # FastAPI-сервис

````

---

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
git clone https://github.com/your-org/MIFIHackatonSberAutoSubscriptionAnalysis.git
cd MIFIHackatonSberAutoSubscriptionAnalysis
pip install -r requirements.txt
````

### 2. Подготовка данных и модели

1. Убедитесь, что в папке `data/model_data/` лежит файл `catboost_optuna_m1.cbm`.
2. Если у вас есть предобработчик признаков, сохранённый в `data/processed_data/data_processed.pkl`, поместите его по указанному пути.

### 3. Запуск API

```bash
uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```

* **GET** `/health` — проверка статуса сервиса
* **POST** `/predict` — предсказание вероятности конверсии

---

## 📦 REST-API

### Запрос `/health`

* **Метод:** GET
* **Ответ:**

  ```json
  { "status": "ok" }
  ```

### Запрос `/predict`

* **Метод:** POST
* **Пример запроса:**

  ```json
  {
  "data": [
    {
      "session_id": "5692861315757623740.1632356796.1632356796",
      "client_id": "2108382700.1637753791",
      "visit_hour": 14,
      "visit_weekday": 2,
      "is_weekend": 0,
      "visit_number": 1,
      "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
      "utm_medium": "banner",
      "utm_campaign": "LEoPHuyFvzoNfnzGgfcd",
      "utm_adcontent": "vCIpmpaGBnIQhyYNkXqp",
      "utm_keyword": "puhZPIYqKXeFPaUviSjo",
      "device_category": "mobile",
      "device_os": "Android",
      "device_brand": "Huawei",
      "device_screen_height": 720,
      "device_screen_width": 360,
      "device_browser": "Chrome",
      "geo_country": "russia",
      "geo_city": "other_russia_city",
      "hit_number": 3,
      "aspect_ratio": 0.5,
      "n_hits": 4,
      "n_target_hits": 1,
      "n_unique_pages": 1,
      "visit_day": 24,
      "visit_month": 11,
      "visit_year": 2021,
      "hit_sec": 3.665,
      "session_total_sec": 42.927
    }
  ]
  }
  ```
* **Пример ответа:**

  ```json
  {
  "session_id": [
    "5692861315757623740.1632356796.1632356796"
  ],
  "client_id": [
    "2108382700.1637753791"
  ],
  "probability": [
    0.053040131394803576
  ]
  }
  ```

---

## 👥 Участники команды

| Роль              | Участник       | Контакты             |
| ----------------- | -------------- | -------------------- |
| Data Engineer     | Анастасия Иванова | @ivnnasti |
| Data Analyst      | Дмитрий Гончаров | @coach_goncharov |
| ML Engineer       | Игнат Шеметов | @Ganyageg |
| Backend Developer | Дмитрий Брагин | @DmitiyBragin |
| Project Manager   | Алексей Сущих | @listaloe |

---

## 🛠 Технологии

* Python 3.8+
* FastAPI, Uvicorn
* Pandas
* CatBoost

---

## 📄 Лицензия

Проект распространяется под лицензией [MIT](LICENSE).