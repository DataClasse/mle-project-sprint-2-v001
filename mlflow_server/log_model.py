import mlflow
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pickle


class MlflowLogger:
    """Утилита для логгирования эксперимента в MLflow."""
    
    def __init__(self, config: Dict[str, Any], run_name: str):
        self.config = config['mlflow']
        mlflow.set_tracking_uri(f"http://{self.config['tracking_server_host']}:{self.config['tracking_server_port']}")
        mlflow.set_experiment(self.config['experiment_name'])
        self.run = mlflow.start_run(run_name=run_name)
        self.stage_name = run_name.split('_')[0]

    def log_params(self, params: Dict[str, Any]):
        """Логирование параметров."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        """Логирование метрик."""
        mlflow.log_metrics(metrics)

    def log_artifacts(self, artifact_paths: List[str]):
        """Логирование артефактов."""
        for path in artifact_paths:
            mlflow.log_artifact(path)

    def log_model(self, model: Any):
        """Логирование и регистрация модели."""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=self.config['model_name']
        )

    def log_dataset(self, data: pd.DataFrame, dataset_name: str = "real_estate_data", 
                   dataset_type: str = "Training"):
        """Логирование dataset в MLflow."""
        try:
            dataset_stats = {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "num_rows": int(len(data)),
                "num_elements": int(data.size),
                "num_columns": int(len(data.columns)),
                "memory_usage_mb": round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "column_names": [str(col) for col in data.columns],
                "dtypes": {str(col): str(data[col].dtype) for col in data.columns}
            }
            
            mlflow.log_dict(dataset_stats, f"{dataset_name}_metadata.json")
            
            sample_data = data.head(100)
            sample_path = f"{dataset_name}_sample.csv"
            sample_data.to_csv(sample_path, index=False)
            mlflow.log_artifact(sample_path)
            
            import os
            os.remove(sample_path)
            
        except Exception as e:
            pass

    def log_data_schema(self, data: pd.DataFrame, dataset_name: str = "real_estate_data"):
        """Логирование схемы данных."""
        try:
            schema_data = {
                "columns": [
                    {
                        "name": col,
                        "type": str(data[col].dtype),
                        "null_count": int(data[col].isnull().sum()),
                        "unique_count": int(data[col].nunique())
                    }
                    for col in data.columns
                ]
            }
            
            mlflow.log_dict(schema_data, f"{dataset_name}_schema.json")
            
        except Exception as e:
            pass

    def log_tags(self, tags: Dict[str, str]):
        """Логирование тегов в MLflow."""
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            pass

    def log_stage_tags(self, stage_name: str, model_type: str = "XGBoost", 
                       dataset_info: Dict[str, Any] = None):
        """Автоматическое логирование тегов для этапа."""
        tags = {
            "stage": stage_name,
            "project": self.config.get('experiment_name', 'unknown'),
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "mlflow_version": mlflow.__version__
        }
        
        if dataset_info:
            tags.update({
                "dataset_rows": str(dataset_info.get('num_rows', 'unknown')),
                "dataset_features": str(dataset_info.get('num_columns', 'unknown'))
            })
        
        self.log_tags(tags)

    def save_stage_results(self, metrics: Dict[str, float], params: Dict[str, Any], 
                          features_info: Dict[str, Any] = None, execution_time: float = None):
        """Сохранение результатов этапа в cv_results/ для сравнения."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.stage_name}_results_{timestamp}.json"
        
        results_data = {
            "stage": self.stage_name,
            "timestamp": datetime.now().isoformat(),
            "model_params": params,
            "metrics": metrics,
            "execution_time": execution_time,
            "mlflow_run_id": self.run.info.run_id if self.run else None
        }
        
        if features_info:
            results_data["features"] = features_info
            
        cv_results_dir = Path("cv_results")
        cv_results_dir.mkdir(exist_ok=True)
        
        results_path = cv_results_dir / filename
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            
        return str(results_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()