# Анализ эффективности сайта «СберАвтоподписка»

Проект направлен на прогнозирование целевых действий пользователей (оставление заявки, заказ звонка и др.) для повышения конверсии сайта.

---

## 📁 Структура репозитория
```
MIFIHackatonSberAutoSubscriptionAnalysis/
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

### 📝 Подробная инструкция для участников проекта

---

#### **1. Установите Git**  
**Для Windows:**  
1. Скачайте Git с [официального сайта](https://git-scm.com/downloads).  
2. Запустите установщик. Оставьте все настройки по умолчанию.  
3. Проверьте установку:  
   ```bash
   git --version
   ```

**Для macOS:**  
1. Установите через Homebrew:  
   ```bash
   brew install git
   ```  
2. Или скачайте с [официального сайта](https://git-scm.com/downloads).  
3. Проверьте установку:  
   ```bash
   git --version
   ```

---

#### **2. Настройте SSH-ключ (для работы с GitHub)**  
1. Сгенерируйте SSH-ключ:  
   ```bash
   ssh-keygen -t ed25519 -C "ваш.email@example.com"
   ```  
   - Нажмите `Enter` для сохранения в стандартную папку.  
   - Парольная фраза — опционально.  

2. Скопируйте публичный ключ:  
   **Windows:**  
   ```bash
   cat ~/.ssh/id_ed25519.pub | clip
   ```  
   **macOS:**  
   ```bash
   pbcopy < ~/.ssh/id_ed25519.pub
   ```  

3. Добавьте ключ в GitHub:  
   - Перейдите в **Settings → SSH and GPG keys → New SSH Key**.  
   - Вставьте ключ из буфера обмена.  

4. Проверьте подключение:  
   ```bash
   ssh -T git@github.com
   ```  
   Ожидаемый вывод: `Hi ваш_логин! You've successfully authenticated...`.

---

#### **3. Клонируйте репозиторий**  
1. Перейдите в папку, где будет храниться проект:  
   ```bash
   cd ~/projects
   ```  
2. Склонируйте репозиторий:  
   ```bash
   git clone git@github.com:listaloe/MIFIHackatonSberAutoSubscriptionAnalysis.git
   ```  

---

#### **4. Настройте локальный репозиторий**  
1. Укажите имя и email (должны совпадать с GitHub):  
   ```bash
   git config --global user.name "Ваше Имя"
   git config --global user.email "ваш.email@example.com"
   ```  

2. Перейдите в папку проекта:  
   ```bash
   cd MIFIHackatonSberAutoSubscriptionAnalysis
   ```  

---

#### **5. Работа с ветками**  
1. Создайте новую ветку для задачи:  
   ```bash
   git checkout -b feature/ваша-фича
   ```  
   Пример: `feature/add-login-form`.  

2. Внесите изменения в код.  

3. Добавьте файлы в коммит:  
   ```bash
   git add .
   ```  

4. Зафиксируйте изменения:  
   ```bash
   git commit -m "Описание изменений"
   ```  

5. Отправьте ветку на GitHub:  
   ```bash
   git push origin feature/ваша-фича
   ```  

---

#### **6. Создайте Pull Request (PR)**  
1. Перейдите на GitHub в ваш репозиторий.  
2. Нажмите **Compare & Pull Request** рядом с вашей веткой.  
3. Заполните описание:  
   - Что сделано?  
   - Как проверить?  
4. Нажмите **Create Pull Request**.  

---

#### **7. Синхронизация с основной веткой**  
1. Переключитесь на ветку `main`:  
   ```bash
   git checkout main
   ```  
2. Заберите изменения из удаленного репозитория:  
   ```bash
   git pull origin main
   ```  
3. Обновите вашу ветку:  
   ```bash
   git checkout feature/ваша-фича
   git merge main
   ```  

---

### 🔧 Частые проблемы и решения  
- **Ошибка «Permission denied»**:  
  Убедитесь, что SSH-ключ добавлен в GitHub и активирован.  

- **Конфликты при слиянии**:  
  Разрешите конфликты вручную в файлах, затем:  
  ```bash
  git add .
  git commit -m "Fix merge conflicts"
  ```  

- **Файлы не добавляются в Git**:  
  Проверьте `.gitignore` — возможно, они там указаны.  

---

### 💡 Советы  
- Всегда делайте `git pull` перед началом работы.  
- Называйте ветки понятно: `fix/button-color`, `docs/update-readme`.  
- Комментируйте изменения подробно: не «Исправлено», а «Исправлено отображение кнопки на мобильных устройствах».  

🚀 **Готово!** Теперь вы можете участвовать в проекте.
