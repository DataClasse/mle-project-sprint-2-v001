#!/usr/bin/env python3
"""
Анализ эффективности замены датасета для register_model_mlflow.py
Конкретное сравнение для задач MLflow регистрации модели
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
import time
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_config():
    """Загружает конфигурацию из config.yaml"""
    config_path = "../config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_data_for_mlflow(data_path, config):
    """Подготавливает данные точно как в register_model_mlflow.py"""
    
    print(f"📥 Загружаем датасет: {data_path}")
    df = pd.read_csv(data_path)
    
    # Получаем целевую переменную из конфигурации
    target_col = config['preprocessing']['features']['target_column']
    
    if target_col not in df.columns:
        raise ValueError(f"Целевая переменная '{target_col}' не найдена в данных")
    
    # Подготавливаем признаки (исключаем целевую переменную)
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Обрабатываем категориальные переменные (как в оригинальном скрипте)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Обрабатываем пропуски
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    return X, y, df

def train_model_like_mlflow_script(X, y, config, dataset_name):
    """Обучает модель точно как в register_model_mlflow.py"""
    
    print(f"\n🤖 ТРЕНИРОВКА МОДЕЛИ ДЛЯ MLFLOW РЕГИСТРАЦИИ: {dataset_name}")
    print("=" * 60)
    
    # Параметры модели из конфигурации (как в оригинальном скрипте)
    model_params = config.get('model', {}).get('params', {})
    random_state = config.get('model', {}).get('random_state', 42)
    test_size = config.get('model', {}).get('test_size', 0.2)
    
    # Добавляем random_state
    model_params['random_state'] = random_state
    
    print(f"📊 Общий размер данных: {X.shape[0]:,} записей")
    print(f"📊 Количество признаков: {X.shape[1]}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 Размер обучающей выборки: {X_train.shape[0]:,}")
    print(f"📊 Размер тестовой выборки: {X_test.shape[0]:,}")
    
    # Создание модели с параметрами из конфигурации
    print(f"⚙️  Параметры модели: {model_params}")
    model = xgb.XGBRegressor(**model_params)
    
    # Замер времени обучения
    start_time = time.time()
    print("⏳ Обучение модели...")
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Метрики (как в MLflow)
    metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'training_time': training_time
    }
    
    # Вычисляем дополнительные метрики для MLflow анализа
    overfit_ratio = metrics['train_rmse'] / metrics['test_rmse'] if metrics['test_rmse'] > 0 else 0
    metrics['overfitting_ratio'] = overfit_ratio
    
    # Информация для MLflow регистрации
    mlflow_info = {
        'model_size_mb': len(pickle.dumps(model)) / 1024 / 1024,
        'feature_count': X.shape[1],
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'feature_names': list(X.columns)
    }
    
    # Показываем результаты
    print(f"\n📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print(f"   ⏱️  Время обучения: {training_time:.2f} сек")
    print(f"   🎯 Train RMSE: {metrics['train_rmse']:,.0f}")
    print(f"   🎯 Test RMSE:  {metrics['test_rmse']:,.0f}")
    print(f"   📊 Train R²:   {metrics['train_r2']:.4f}")
    print(f"   📊 Test R²:    {metrics['test_r2']:.4f}")
    print(f"   ⚖️  Переобучение: {overfit_ratio:.2f}x ({'🟢 Низкое' if overfit_ratio < 1.2 else '🟡 Среднее' if overfit_ratio < 2.0 else '🔴 Высокое'})")
    
    return model, metrics, mlflow_info

def analyze_mlflow_compatibility(data_path, dataset_name):
    """Анализирует совместимость с MLflow pipeline"""
    
    print(f"\n🔍 АНАЛИЗ СОВМЕСТИМОСТИ С MLFLOW: {dataset_name}")
    print("=" * 60)
    
    # Проверяем размер файла
    file_size_mb = os.path.getsize(data_path) / 1024 / 1024
    print(f"📁 Размер файла: {file_size_mb:.1f} MB")
    
    # Загружаем данные
    df = pd.read_csv(data_path)
    
    # Анализ для MLflow
    mlflow_analysis = {
        'file_size_mb': file_size_mb,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'missing_values': df.isnull().sum().sum(),
        'target_range': (df['price'].min(), df['price'].max()) if 'price' in df.columns else (0, 0),
        'target_mean': df['price'].mean() if 'price' in df.columns else 0
    }
    
    print(f"💾 Использование памяти: {mlflow_analysis['memory_usage_mb']:.1f} MB")
    print(f"📊 Записи: {mlflow_analysis['rows']:,}")
    print(f"📋 Столбцы: {mlflow_analysis['columns']}")
    print(f"🎯 Диапазон цен: {mlflow_analysis['target_range'][0]:,.0f} - {mlflow_analysis['target_range'][1]:,.0f}")
    print(f"📈 Средняя цена: {mlflow_analysis['target_mean']:,.0f}")
    
    # Оценка влияния на MLflow
    print(f"\n📋 ВЛИЯНИЕ НА MLFLOW ПРОЦЕССЫ:")
    
    # Время загрузки данных
    start_time = time.time()
    _ = pd.read_csv(data_path)
    load_time = time.time() - start_time
    print(f"   ⏱️  Время загрузки: {load_time:.2f} сек")
    
    # Оценка размера артефактов MLflow
    estimated_model_size = mlflow_analysis['rows'] * mlflow_analysis['columns'] * 0.001  # примерная оценка в MB
    print(f"   📦 Ожидаемый размер модели: ~{estimated_model_size:.1f} MB")
    
    # Рекомендации для MLflow
    recommendations = []
    if file_size_mb > 50:
        recommendations.append("⚠️  Большой размер файла - рассмотреть сжатие")
    if mlflow_analysis['rows'] > 500000:
        recommendations.append("⚠️  Много записей - может потребоваться больше памяти")
    if mlflow_analysis['missing_values'] > 0:
        recommendations.append("⚠️  Есть пропущенные значения")
    
    if recommendations:
        print(f"\n🔧 РЕКОМЕНДАЦИИ ДЛЯ MLFLOW:")
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print(f"\n✅ ОПТИМАЛЬНО ДЛЯ MLFLOW")
    
    mlflow_analysis['load_time'] = load_time
    mlflow_analysis['estimated_model_size'] = estimated_model_size
    mlflow_analysis['recommendations'] = recommendations
    
    return mlflow_analysis

def compare_for_mlflow_registration(original_path, aligned_path, config):
    """Основное сравнение для MLflow регистрации"""
    
    print("🆚 СРАВНЕНИЕ ДАТАСЕТОВ ДЛЯ MLFLOW РЕГИСТРАЦИИ")
    print("=" * 70)
    
    results = {}
    
    # Анализ совместимости
    print("\n🔍 АНАЛИЗ СОВМЕСТИМОСТИ С MLFLOW ПРОЦЕССАМИ")
    results['original_mlflow'] = analyze_mlflow_compatibility(original_path, "ИСХОДНЫЙ")
    results['aligned_mlflow'] = analyze_mlflow_compatibility(aligned_path, "ПРИВЕДЕННЫЙ")
    
    # Подготовка данных
    print(f"\n📊 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    X_orig, y_orig, df_orig = prepare_data_for_mlflow(original_path, config)
    X_aligned, y_aligned, df_aligned = prepare_data_for_mlflow(aligned_path, config)
    
    # Обучение моделей
    model_orig, metrics_orig, mlflow_orig = train_model_like_mlflow_script(X_orig, y_orig, config, "ИСХОДНЫЙ")
    model_aligned, metrics_aligned, mlflow_aligned = train_model_like_mlflow_script(X_aligned, y_aligned, config, "ПРИВЕДЕННЫЙ")
    
    results['original_metrics'] = metrics_orig
    results['aligned_metrics'] = metrics_aligned
    results['original_mlflow_info'] = mlflow_orig
    results['aligned_mlflow_info'] = mlflow_aligned
    
    return results

def generate_mlflow_recommendation(results):
    """Генерирует рекомендацию специально для MLflow регистрации"""
    
    print(f"\n🎯 АНАЛИЗ ЭФФЕКТИВНОСТИ ДЛЯ REGISTER_MODEL_MLFLOW.PY")
    print("=" * 70)
    
    orig_metrics = results['original_metrics']
    aligned_metrics = results['aligned_metrics']
    orig_mlflow = results['original_mlflow']
    aligned_mlflow = results['aligned_mlflow']
    
    # Таблица сравнения ключевых метрик для MLflow
    print(f"\n📊 СРАВНЕНИЕ КЛЮЧЕВЫХ МЕТРИК ДЛЯ MLFLOW:")
    print(f"{'Метрика':<25} {'Исходный':<15} {'Приведенный':<15} {'Изменение':<15}")
    print("-" * 75)
    
    key_metrics = [
        ('Test RMSE', 'test_rmse'),
        ('Test R²', 'test_r2'),
        ('Время обучения (сек)', 'training_time'),
        ('Переобучение (ratio)', 'overfitting_ratio')
    ]
    
    improvements = {}
    
    for display_name, metric_key in key_metrics:
        orig_val = orig_metrics[metric_key]
        aligned_val = aligned_metrics[metric_key]
        
        if metric_key in ['test_rmse', 'training_time', 'overfitting_ratio']:
            # Меньше = лучше
            change = ((orig_val - aligned_val) / orig_val) * 100
            is_better = aligned_val < orig_val
        else:
            # Больше = лучше
            change = ((aligned_val - orig_val) / abs(orig_val)) * 100
            is_better = aligned_val > orig_val
        
        improvements[metric_key] = change
        status = "🟢" if is_better else "🔴"
        
        print(f"{display_name:<25} {orig_val:<15.3f} {aligned_val:<15.3f} {status} {change:+.1f}%")
    
    # Анализ влияния на MLflow процессы
    print(f"\n🔧 ВЛИЯНИЕ НА MLFLOW ПРОЦЕССЫ:")
    print(f"{'Аспект':<25} {'Исходный':<15} {'Приведенный':<15} {'Статус'}")
    print("-" * 70)
    
    mlflow_aspects = [
        ('Размер файла (MB)', orig_mlflow['file_size_mb'], aligned_mlflow['file_size_mb']),
        ('Время загрузки (сек)', orig_mlflow['load_time'], aligned_mlflow['load_time']),
        ('Записей для обучения', orig_mlflow['training_samples'], aligned_mlflow['training_samples']),
        ('Размер модели (MB)', orig_mlflow['model_size_mb'], aligned_mlflow['model_size_mb'])
    ]
    
    for aspect, orig_val, aligned_val in mlflow_aspects:
        if aspect == 'Записей для обучения':
            status = "🟢" if aligned_val > orig_val else "🔴"
        else:
            status = "🟡" if abs(aligned_val - orig_val) / orig_val < 0.5 else "🔴" if aligned_val > orig_val * 1.5 else "🟢"
        
        print(f"{aspect:<25} {orig_val:<15.1f} {aligned_val:<15.1f} {status}")
    
    # Рекомендации для MLflow регистрации
    print(f"\n🎯 РЕКОМЕНДАЦИЯ ДЛЯ REGISTER_MODEL_MLFLOW.PY:")
    
    # Критерии оценки
    better_quality = improvements['test_r2'] > 5  # R² улучшился на 5%+
    acceptable_rmse = improvements['test_rmse'] > -20  # RMSE ухудшился не более чем на 20%
    no_severe_overfitting = aligned_metrics['overfitting_ratio'] < 3  # Переобучение не критичное
    more_data = aligned_mlflow['training_samples'] > orig_mlflow['training_samples']
    
    criteria_met = sum([better_quality, acceptable_rmse, no_severe_overfitting, more_data])
    
    print(f"\n📋 КРИТЕРИИ ОЦЕНКИ:")
    print(f"   {'✅' if better_quality else '❌'} Качество модели (R²): {'Улучшилось' if better_quality else 'Не улучшилось значительно'}")
    print(f"   {'✅' if acceptable_rmse else '❌'} Точность (RMSE): {'Приемлемо' if acceptable_rmse else 'Значительно ухудшилась'}")
    print(f"   {'✅' if no_severe_overfitting else '❌'} Переобучение: {'Контролируемо' if no_severe_overfitting else 'Критичное'}")
    print(f"   {'✅' if more_data else '❌'} Объем данных: {'Увеличился' if more_data else 'Остался прежним'}")
    
    if criteria_met >= 3:
        recommendation = "🟢 РЕКОМЕНДУЕТСЯ ЗАМЕНА"
        reasoning = "Приведенный датасет показывает лучшие результаты для MLflow регистрации"
    elif criteria_met >= 2:
        recommendation = "🟡 УСЛОВНАЯ ЗАМЕНА"
        reasoning = "Есть улучшения, но требуется дополнительная настройка модели"
    else:
        recommendation = "🔴 НЕ РЕКОМЕНДУЕТСЯ ЗАМЕНА"
        reasoning = "Приведенный датасет не показывает значительных преимуществ"
    
    print(f"\n{recommendation}")
    print(f"📝 Обоснование: {reasoning}")
    print(f"🎯 Критериев выполнено: {criteria_met}/4")
    
    # Конкретные шаги для register_model_mlflow.py
    print(f"\n🔧 КОНКРЕТНЫЕ ШАГИ ДЛЯ REGISTER_MODEL_MLFLOW.PY:")
    
    if "РЕКОМЕНДУЕТСЯ" in recommendation:
        print(f"   1. ✅ Обновить config.yaml:")
        print(f"      initial_data: 'data/merged_data_improved_aligned.csv'")
        print(f"   2. ✅ Запустить register_model_mlflow.py")
        print(f"   3. ✅ Сравнить в MLflow UI с Run ID: a325fef21dad44c396b49a0b63cee154")
        print(f"   4. ✅ При необходимости настроить гиперпараметры")
    else:
        print(f"   1. ⚠️  Сначала оптимизировать модель:")
        print(f"      - Добавить регуляризацию (reg_alpha, reg_lambda)")
        print(f"      - Уменьшить max_depth и увеличить min_child_weight")
        print(f"   2. ⚠️  Затем повторить анализ")
    
    return recommendation, criteria_met

def main():
    """Основная функция анализа"""
    
    # Пути к датасетам
    original_path = "data/initial_data_set.csv"
    aligned_path = "data/merged_data_improved_aligned.csv"
    
    # Проверяем наличие файлов
    if not os.path.exists(original_path):
        print(f"❌ Файл не найден: {original_path}")
        return
    
    if not os.path.exists(aligned_path):
        print(f"❌ Файл не найден: {aligned_path}")
        return
    
    # Загружаем конфигурацию
    config = load_config()
    
    # Проводим сравнение
    results = compare_for_mlflow_registration(original_path, aligned_path, config)
    
    # Генерируем рекомендацию
    recommendation, score = generate_mlflow_recommendation(results)
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_results = {
        'timestamp': timestamp,
        'recommendation': recommendation,
        'score': f"{score}/4",
        'comparison_for': 'register_model_mlflow.py',
        'datasets': {
            'original': original_path,
            'aligned': aligned_path
        },
        'metrics': {
            'original': results['original_metrics'],
            'aligned': results['aligned_metrics']
        },
        'mlflow_impact': {
            'original': results['original_mlflow'],
            'aligned': results['aligned_mlflow']
        }
    }
    
    results_path = f"cv_results/mlflow_dataset_comparison_{timestamp}.json"
    os.makedirs("cv_results", exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        # Используем специальный encoder для numpy types
        json.dump(output_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Результаты сохранены: {results_path}")
    
    return recommendation, results

if __name__ == "__main__":
    try:
        recommendation, results = main()
        print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН")
        print(f"🎯 ИТОГОВАЯ РЕКОМЕНДАЦИЯ: {recommendation}")
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        raise
