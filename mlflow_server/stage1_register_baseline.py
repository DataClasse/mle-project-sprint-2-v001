import yaml
import pickle
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from log_model import MlflowLogger
from dotenv import load_dotenv

def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    """Загрузка конфигурации проекта."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def register_baseline():
    """Основная функция для регистрации baseline модели."""
    
    # 1. Загрузка конфига
    config = load_config()
    
    # 2. Загрузка переменных окружения из .env
    env_path = Path(config['paths']['env_file'])
    if env_path.exists():
        load_dotenv(env_path)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    
    # 3. Загрузка готовой модели и метрик
    model_path = Path(config['paths']['data']['model_path']) / "price_prediction_baseline_model.pkl"
    metrics_path = Path(config['paths']['results']['cv_results']) / "stage1_results_20250817_152404.json"

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # 4. Загрузка данных для логирования dataset
    data_path = Path(config['paths']['data']['initial'])
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        data = None

    # 5. Использование Mlflow_logger для логирования
    with MlflowLogger(config, "stage1_baseline_registration") as logger:
        
        # Логирование тегов этапа
        logger.log_stage_tags(
            stage_name="stage1_baseline_registration",
            model_type="XGBoost",
            dataset_info={"num_rows": data.shape[0], "num_columns": data.shape[1]} if data is not None else None
        )
        
        # Логирование дополнительных тегов
        logger.log_tags({
            "baseline": "true",
            "production_ready": "false",
            "business_value": "high",
            "target_column": "price",
            "problem_type": "regression",
            "domain": "real_estate"
        })
        
        # Логирование параметров baseline модели
        logger.log_params(config['model']['params'])
        
        try:
            # Логирование метрик (только числовые значения)
            metrics_to_log = metrics['metrics']
            logger.log_metrics(metrics_to_log)
            
            # Логирование dataset (если данные загружены)
            if data is not None:
                logger.log_dataset(data, "real_estate_data", "Training")
                logger.log_data_schema(data, "real_estate_data")
            
            # Логирование артефактов
            logger.log_artifacts(["../config.yaml", str(metrics_path)])
            
            # Логирование модели
            logger.log_model(model)
            
            # Сохранение результатов этапа в cv_results/
            logger.save_stage_results(metrics, config['model']['params'])
            
        except Exception as e:
            # Сохраняем результаты локально даже при ошибке MLflow
            logger.save_stage_results(metrics, config['model']['params'])

if __name__ == "__main__":

    register_baseline()