#!/usr/bin/env python3
"""Этап 1: Регистрация baseline модели в MLflow"""

import json
import os

import joblib
import mlflow
import mlflow.catboost
import pandas as pd
import yaml
from dotenv import load_dotenv


def main():
    """Основная функция для регистрации baseline модели."""
    # Настройка MLflow
    project_root = os.path.dirname(os.path.dirname(__file__))
    load_dotenv(os.path.join(project_root, ".env"))

    with open(os.path.join(project_root, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mlflow_config = config.get("mlflow", {})
    mlflow.set_tracking_uri(
        f"http://{mlflow_config.get('tracking_server_host', 'localhost')}:{mlflow_config.get('tracking_server_port', 5000)}"
    )
    mlflow.set_experiment(mlflow_config.get("experiment_name", "Model Development"))

    # Регистрация модели
    with mlflow.start_run(run_name="stage_1_baseline_registration"):
        # Теги
        mlflow.set_tags(
            {
                "stage": "stage_1_baseline",
                "model_type": "CatBoost",
                "purpose": "baseline_registration",
                "version": "v1.0",
            }
        )

        # Загрузка модели и метрик
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        with open(
            os.path.join(models_dir, "price_prediction_baseline_model.pkl"), "rb"
        ) as f:
            model_data = joblib.load(f)
        model = model_data.get("model", model_data)

        with open(os.path.join(models_dir, "baseline_metrics.json"), "r") as f:
            metrics = json.load(f)

        # Логирование метрик (только числовые)
        numeric_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        mlflow.log_metrics(numeric_metrics)

        # Логирование параметров
        feature_names = metrics.get("feature_names", [])
        mlflow.log_param("feature_names_count", len(feature_names))
        mlflow.log_param("feature_names", str(feature_names))

        # Безопасное получение параметров модели
        try:
            model_params = model.get_params()
        except AttributeError:
            model_params = {
                "iterations": getattr(model, "iterations", 100),
                "learning_rate": getattr(model, "learning_rate", 0.1),
                "depth": getattr(model, "depth", 6),
                "loss_function": getattr(model, "loss_function", "RMSE"),
                "verbose": getattr(model, "verbose", False),
                "random_seed": getattr(model, "random_seed", None),
            }
        mlflow.log_params(model_params)

        # Создание сигнатуры и логирование модели
        input_data = pd.read_csv(os.path.join(models_dir, "training_data_sample.csv"))
        prediction = model.predict(input_data)
        signature = mlflow.models.infer_signature(input_data, prediction)

        # Логирование артефактов
        mlflow.log_artifact(
            os.path.join(project_root, "config.yaml"), "project_config.yaml"
        )
        mlflow.log_artifact(
            os.path.join(project_root, "requirements.txt"),
            "environment_requirements.txt",
        )

        # Временный файл для артефакта
        temp_file = "training_data_sample.csv"
        input_data.to_csv(temp_file, index=False)
        mlflow.log_artifact(temp_file, "training_data_sample.csv")
        os.remove(temp_file)

        # Регистрация модели
        mlflow.catboost.log_model(
            model,
            "baseline_model",
            registered_model_name="price_prediction_model",
            signature=signature,
            input_example=input_data.head(10),
        )


if __name__ == "__main__":
    main()
