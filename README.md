# Анализ эффективности сайта «СберАвтоподписка»

Проект направлен на прогнозирование целевых действий пользователей (оставление заявки, заказ звонка и др.) для повышения конверсии сайта.

---

## 📁 Структура репозитория
```
sberautopodpiska-analysis/
├── data/                    # Исходные данные (игнорируются Git)
├── notebooks/               # Jupyter-ноутбуки
│   ├── data_preprocessing/  # Очистка и подготовка данных
│   ├── EDA/                 # Разведочный анализ
│   └── modeling/            # Эксперименты с моделями
├── src/                     # Исходный код
│   ├── data_processing/     # Скрипты для обработки данных
│   ├── models/              # Обучение и валидация моделей
│   └── api/                 # API для предсказаний (Flask/FastAPI)
├── tests/                   # Тесты
├── docs/                    # Документация
├── scripts/                 # Вспомогательные скрипты
└── .github/                 # Настройки CI/CD
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
git clone https://github.com/listaloe/MIFIHackatonSberAutoSubscriptionAnalysis.git
cd MIFIHackatonSberAutoSubscriptionAnalysis
pip install -r requirements.txt
```

### 2. Запуск API
```bash
cd src/api
python app.py
```
**Пример запроса:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "utm_source": "google",
    "device_category": "mobile",
    "geo_city": "Москва"
  }'
```

---

## 📊 Данные
Данные хранятся в закрытом доступе. Для работы используйте синтетические примеры из папки `data/sample/`.

---

## 👥 Участники команды
| Роль | Участник | Контакты |
|------|----------|----------|
| Data Engineer | [Имя] | [@телеграм/почта] |
| Data Analyst | [Имя] | [@телеграм/почта] |
| ML Engineer | [Имя] | [@телеграм/почта] |
| Backend Developer | [Имя] | [@телеграм/почта] |

---

## 📝 Глоссарий
- **Целевое действие**: `ga_hits.event_action` из списка: `["оставить_заявку", "заказать_звонок"]`.
- **CR (Conversion Rate)**: Доля визитов с целевыми действиями.
- **Органический трафик**: `utm_medium in ('organic', 'referral', '(none)')`.

---

## 🛠 Технологии
- Python 3.9+
- Scikit-learn, LightGBM
- Flask, Docker
- Pandas, Matplotlib

---

## 📄 Лицензия
Проект распространяется под лицензией [MIT](LICENSE).

---

## 🤝 Как внести вклад
1. Создайте ветку: `git checkout -b feature/your-feature`.
2. Зафиксируйте изменения: `git commit -m "Описание фичи"`.
3. Запушьте ветку: `git push origin feature/your-feature`.
4. Создайте Pull Request.
