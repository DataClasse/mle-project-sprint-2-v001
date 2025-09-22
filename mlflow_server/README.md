# ЭТАП 1: BASELINE МОДЕЛЬ В MLFLOW

## ВЫПОЛНЕНИЕ

```bash
cd mle_projects/mle-project-sprint-2-v001/mlflow_server/

# Запуск MLflow сервера
bash rms.sh &

# Регистрация модели в MLflow
python3 stage_1_register_baseline.py
```

## РЕЗУЛЬТАТЫ

- ✅ **Модель**: `price_prediction_model` (версия 55)
- ✅ **Эксперимент**: `baseline_model_improvement_REPP`
- ✅ **Test RMSE**: 4,157,670 ₽
- ✅ **Test R²**: 0.880
- ✅ **Test MAE**: 2,399,175 ₽
- ✅ **CV RMSE**: 4,370,851 ± 148,409 ₽
- ✅ **Данные**: 120,118 записей (15 признаков)

## ДОСТУП

- **MLflow UI**: http://127.0.0.1:5000
