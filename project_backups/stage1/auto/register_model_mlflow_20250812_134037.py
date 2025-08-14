#!/usr/bin/env python3
import yaml
import json
import os
import pickle
import glob
import logging
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import pandas as pd
import xgboost as xgb

def load_config():
    """Загружает конфигурацию из config.yaml"""
    config_path = "../config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_env():
    """Загружает переменные окружения из .env"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f"Файл .env не найден по пути: {env_path}")

def setup_environment():
    """Настраивает переменные окружения для MLflow"""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not os.getenv("S3_BUCKET_NAME"):
        os.environ["S3_BUCKET_NAME"] = "mlflow-artifacts"

def setup_logging(config):
    """Настраивает логирование из конфигурации"""
    logging_config = config.get('logging', {})
    
    log_level = getattr(logging, logging_config.get('level', 'INFO'))
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file', 'logs/mlflow_registration.log')
    
    # Создаём директорию для логов если не существует
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def log_xgboost_model():
    """Регистрирует XGBoost модель в MLflow"""
    
    # Загружаем переменные окружения из .env файла
    load_env()
    
    # Настраиваем окружение
    setup_environment()
    
    # Загружаем конфигурацию
    config = load_config()
    
    # Настраиваем логирование
    logger = setup_logging(config)
    logger.info("Начинаем регистрацию модели в MLflow")
    
    # Настройки MLflow из конфигурации
    mlflow_config = config['mlflow']
    experiment_name = mlflow_config['experiment_name']
    model_name = mlflow_config['model_name']
    
    # Устанавливаем URI MLflow
    tracking_uri = f"http://{mlflow_config['tracking_server_host']}:{mlflow_config['tracking_server_port']}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    
    # Настраиваем S3 хранилище артефактов
    s3_bucket = os.getenv("S3_BUCKET_NAME", "mlflow-artifacts")
    artifact_location = f"s3://{s3_bucket}/mlflow-artifacts"
   
    # Пути к файлам из конфигурации
    saving_config = config['saving']
    models_dir = saving_config['models_dir']
    cv_results_dir = saving_config['cv_results_dir']
    
    # Путь к данным из конфигурации
    data_path = Path(__file__).parent / config['automation']['data_paths']['initial_data']
    
    # Находим актуальные файлы модели и метрик
    model_path = f"{models_dir}/price_prediction_baseline_model.pkl"
    
    # Ищем последние файлы метрик и признаков
    metrics_files = glob.glob(f"{cv_results_dir}/xgboost_metrics_*.json")
    features_files = glob.glob(f"{cv_results_dir}/xgboost_features_*.json")
    
    if not metrics_files or not features_files:
        raise FileNotFoundError(f"Не найдены файлы метрик или признаков в {cv_results_dir}")
    
    # Используем последние файлы (по времени модификации)
    metrics_path = max(metrics_files, key=os.path.getmtime)
    features_path = max(features_files, key=os.path.getmtime)
    
    # Проверяем существование файлов
    required_files = [model_path, metrics_path, features_path, str(data_path)]
    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    # Загружаем метрики (CV-результаты для логирования в MLflow)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Загружаем информацию о признаках
    with open(features_path, 'r') as f:
        features_info = json.load(f)
    
    # Загружаем обученную модель
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Загружаем готовые данные
    logger.info(f"Загружаем предобработанные данные: {data_path}")
    data = pd.read_csv(str(data_path))
    
    # Получаем параметры модели
    model_params = model.get_params()
    
    # Используем только те признаки, которые использовались при обучении
    expected_features = features_info['feature_names']
    
    # Проверяем, что все ожидаемые признаки есть в данных
    missing_features = [f for f in expected_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Отсутствуют признаки в данных: {missing_features}")
    
    # Подготавливаем данные для подписи модели (только нужные признаки)
    features_data = data[expected_features].copy()
    
    # Создаем/используем существующий эксперимент MLflow
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(
            experiment_name, 
            artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
    
    # Создаем подпись модели
    prediction = model.predict(features_data[:10])  # Пример для 10 записей
    signature = mlflow.models.infer_signature(features_data[:10], prediction)
    input_example = features_data[:5]  # Пример входных данных
    
    # Начинаем MLflow run с именем из конфигурации
    from datetime import datetime
    automation_config = config['automation']
    timestamp = datetime.now().strftime(saving_config['timestamp_format'])
    run_name = automation_config['run_naming'].format(timestamp=timestamp)
    logger.info(f"Создаём MLflow run: {run_name}")
    
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        
        # Логируем основные параметры модели XGBoost
        important_params = {k: v for k, v in model_params.items() if v is not None}
        
        key_params = {
            'objective': important_params.get('objective'),
            'booster': important_params.get('booster'),
            'base_score': important_params.get('base_score'),
            'missing': important_params.get('missing'),
            'enable_categorical': important_params.get('enable_categorical')
        }
        
        # Убираем None значения
        key_params = {k: v for k, v in key_params.items() if v is not None}
        
        # Логируем параметры модели
        mlflow.log_params(key_params)
        logger.info(f"Залогированы параметры модели: {len(key_params)} штук")
        
        # Условное логирование метрик
        artifacts_config = config['artifacts']
        if artifacts_config.get('metrics', True):
            mlflow.log_metrics(metrics)
            logger.info(f"Залогированы метрики: {list(metrics.keys())}")
        
        # Логируем только ключевые метаданные как теги (убираем дублирование)
        key_tags = {
            "dataset_name": "real_estate_data",
            "dataset_version": "1.0",
            "dataset_target": config['preprocessing']['features']['target_column'],
            "dataset_features_count": len(expected_features),
            "dataset_records_count": len(data),
            "dataset_preprocessing": "completed",
            "dataset_size_mb": f"{data.memory_usage(deep=True).sum() / 1024**2:.1f}"
        }
        
        # Логируем оптимизированные теги
        for key, value in key_tags.items():
            mlflow.set_tag(key, str(value))
        
        logger.info(f"Созданы оптимизированные метаданные Dataset: {key_tags['dataset_name']} v{key_tags['dataset_version']}")
        
        # Логируем dataframe как dataset для отображения в разделе Datasets
        try:
            # Создаём dataset из pandas DataFrame
            dataset = mlflow.data.from_pandas(
                data, 
                source=str(data_path),
                name="real_estate_data"
            )
            mlflow.log_input(dataset, context="training")
            logger.info(f"Dataframe успешно залогирован как dataset: {dataset.name}")
        except Exception as e:
            logger.warning(f"Не удалось создать dataset, используем fallback: {e}")
            # Fallback: логируем как артефакт
            mlflow.log_artifact(str(data_path), "dataframe")
            logger.info(f"Залогированы данные как артефакт (fallback): {data_path}")
        
        # Логируем модель XGBoost с pip_requirements и artifact_path - обязательные элементы
        script_dir = Path(__file__).parent
        pip_requirements = str(script_dir / "requirements_baseline.txt")
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="models",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            await_registration_for=60
        )
        
        # Устанавливаем только ключевые теги модели (убираем дублирование)
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("model_name", config['model']['name'])
        mlflow.set_tag("model_params_count", len(config['model']['params']))
        
        logger.info(f"Модель успешно зарегистрирована в MLflow. Run ID: {run_id}")
        logger.info(f"Model URI: {model_info.model_uri}")
        
        return model_info

if __name__ == "__main__":
    try:
        model_info = log_xgboost_model()
        print(f"✅ Регистрация модели завершена успешно!")
        print(f"Model URI: {model_info.model_uri}")
    except Exception as e:
        print(f"❌ Ошибка при регистрации модели: {e}")
        logging.error(f"Ошибка при регистрации модели: {e}", exc_info=True)
        raise