# MLflow Server - Этап 1: Регистрация Baseline Модели

## Описание проекта

Проект "Улучшение baseline-модели" для Яндекс Недвижимости. Цель - оптимизировать подход к прогнозированию стоимости недвижимости, используя MLflow для отслеживания экспериментов и обеспечения воспроизводимости результатов.

## Этап 1: Разворачивание MLflow Tracking Server и MLflow Model Registry

### Задача этапа
Развернуть сервисы MLflow Tracking Server и MLflow Model Registry, используя:
- База данных PostgreSQL для хранения метаданных экспериментов
- Объектное хранилище S3 (Yandex Cloud) для артефактов
- Регистрация существующей baseline модели

### Что реализовано

#### 1. MLflow Tracking Server
- **Скрипт запуска**: `rms.sh` - автоматический запуск MLflow сервера
- **Конфигурация**: PostgreSQL как backend-store-uri, S3 как default-artifact-root
- **Переменные окружения**: загрузка из `.env` файла

#### 2. Регистрация Baseline Модели
- **Скрипт**: `stage1_register_baseline.py` - регистрация модели в MLflow
- **Логирование**: параметры, метрики, артефакты, окружение
- **Модель**: XGBoost baseline модель из предыдущего спринта

#### 3. MLflow Logger
- **Утилита**: `log_model.py` - класс для логирования экспериментов
- **Функции**: логирование параметров, метрик, артефактов, моделей
- **Дополнительно**: логирование dataset, схемы данных, тегов

## Структура проекта

```
mlflow_server/
├── data/                    # Данные проекта
├── models/                  # Модели 
├── cv_results/             # Результаты кросс-валидации
├── logs/                   # Логи выполнения
├── log_model.py            # Логгер MLflow
├── stage1_register_baseline.py  # Скрипт этапа 1
├── rms.sh                  # Скрипт запуска MLflow сервера
└── README.md               # Документация
```

## Требования

### Переменные окружения (.env)
```bash
# PostgreSQL для MLflow
DB_DESTINATION_HOST=your_host
DB_DESTINATION_PORT=5432
DB_DESTINATION_NAME=your_db_name
DB_DESTINATION_USER=your_user
DB_DESTINATION_PASSWORD=your_password

# S3 (Yandex Cloud)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name

# MLflow Server
TRACKING_SERVER_HOST=127.0.0.1
TRACKING_SERVER_PORT=5000
```

### Зависимости Python
```bash
pip install mlflow==2.7.1 pandas numpy scikit-learn xgboost python-dotenv pyyaml
```

## Запуск

### 1. Запуск MLflow сервера
```bash
# Сделать скрипт исполняемым
chmod +x rms.sh

# Запустить MLflow сервер
./rms.sh
```

**Что происходит:**
- Загружаются переменные окружения из `.env`
- Проверяется доступность PostgreSQL
- Освобождается порт 5000 (если занят)
- Запускается MLflow сервер с настройками:
  - Backend: PostgreSQL
  - Artifacts: S3 (Yandex Cloud)
  - Host: 127.0.0.1:5000

### 2. Регистрация baseline модели
```bash
python3 stage1_register_baseline.py
```

**Что происходит:**
- Загружается конфигурация из `config.yaml`
- Загружается baseline модель XGBoost
- Загружаются метрики тестовой выборки
- Загружается исходный dataset
- Логируется в MLflow:
  - Параметры модели
  - Метрики качества
  - Dataset и схема данных
  - Артефакты (конфиг, метрики)
  - Модель (регистрируется в Model Registry)

## Результаты этапа

### 1. MLflow Tracking Server
- ✅ Запущен и доступен по адресу http://127.0.0.1:5000
- ✅ Настроена интеграция с PostgreSQL
- ✅ Настроена интеграция с S3 (Yandex Cloud)

### 2. Baseline модель зарегистрирована
- ✅ Параметры модели залогированы
- ✅ Метрики тестовой выборки залогированы
- ✅ Dataset и схема данных залогированы
- ✅ Модель сохранена в MLflow Model Registry
- ✅ Все артефакты доступны в MLflow

### 3. Готовность к следующим этапам
- ✅ MLflow инфраструктура развернута
- ✅ Baseline модель зарегистрирована
- ✅ Система логирования настроена
- ✅ Можно переходить к этапу 2 (EDA анализ)

## Проверка работы

### 1. Проверка MLflow сервера
```bash
# Проверить статус сервера
curl http://127.0.0.1:5000/health

# Открыть веб-интерфейс
open http://127.0.0.1:5000
```

### 2. Проверка регистрации модели
- Открыть MLflow UI
- Перейти в эксперимент "baseline_model_improvement_REPP"
- Найти run "stage1_baseline_registration"
- Проверить залогированные параметры, метрики, артефакты
- Проверить модель в Model Registry

