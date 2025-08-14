#!/usr/bin/env python3
"""
Сравнение влияния исходных и улучшенных данных на параметры модели
Анализ для принятия решения о замене датасета
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
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

def analyze_dataset(df, name):
    """Анализирует структуру и качество датасета"""
    print(f"\n🔍 АНАЛИЗ ДАТАСЕТА: {name}")
    print("=" * 50)
    
    # Базовая информация
    print(f"📊 Размер: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    print(f"💾 Память: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Типы данных
    print(f"\n📋 ТИПЫ ДАННЫХ:")
    data_types = df.dtypes.value_counts()
    for dtype, count in data_types.items():
        print(f"   {dtype}: {count} столбцов")
    
    # Пропущенные значения
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_info = missing[missing > 0]
    
    if len(missing_info) > 0:
        print(f"\n❌ ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
        for col, count in missing_info.items():
            print(f"   {col}: {count:,} ({missing_pct[col]:.1f}%)")
    else:
        print(f"\n✅ Пропущенных значений нет")
    
    # Числовые столбцы
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 ЧИСЛОВЫЕ СТОЛБЦЫ: {len(numeric_cols)}")
        numeric_stats = df[numeric_cols].describe()
        print("   Мин/Макс значения:")
        for col in numeric_cols[:5]:  # Показываем первые 5
            min_val = numeric_stats.loc['min', col]
            max_val = numeric_stats.loc['max', col]
            print(f"   {col}: {min_val:.2f} — {max_val:.2f}")
        if len(numeric_cols) > 5:
            print(f"   ... и ещё {len(numeric_cols) - 5} столбцов")
    
    # Категориальные столбцы
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"\n📝 КАТЕГОРИАЛЬНЫЕ СТОЛБЦЫ: {len(cat_cols)}")
        for col in cat_cols[:3]:  # Показываем первые 3
            unique_count = df[col].nunique()
            print(f"   {col}: {unique_count} уникальных значений")
        if len(cat_cols) > 3:
            print(f"   ... и ещё {len(cat_cols) - 3} столбцов")
    
    return {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_count': len(missing_info),
        'missing_total': missing.sum(),
        'numeric_cols': len(numeric_cols),
        'categorical_cols': len(cat_cols),
        'columns': list(df.columns)
    }

def prepare_data_for_ml(df, config, target_col):
    """Подготавливает данные для машинного обучения"""
    # Убираем target из признаков
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Обрабатываем категориальные переменные
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"📝 Обрабатываем {len(categorical_cols)} категориальных столбцов...")
        from sklearn.preprocessing import LabelEncoder
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Обрабатываем пропуски
    if X.isnull().sum().sum() > 0:
        print("🔧 Заполняем пропущенные значения медианой...")
        X = X.fillna(X.median())
    
    return X, y

def train_and_evaluate_model(X, y, name, config):
    """Обучает модель и возвращает метрики"""
    print(f"\n🤖 ОБУЧЕНИЕ МОДЕЛИ НА: {name}")
    print("=" * 50)
    
    # Разделяем данные
    test_size = config.get('model', {}).get('test_size', 0.2)
    random_state = config.get('model', {}).get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"📊 Размер обучающей выборки: {X_train.shape[0]:,}")
    print(f"📊 Размер тестовой выборки: {X_test.shape[0]:,}")
    
    # Параметры модели из конфигурации
    model_params = config.get('model', {}).get('params', {})
    model_params['random_state'] = random_state
    
    # Создаем и обучаем модель
    model = xgb.XGBRegressor(**model_params)
    
    print("⏳ Обучение модели...")
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Метрики
    metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    # Cross-validation
    print("🔄 Cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
    metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
    
    # Показываем результаты
    print(f"\n📈 РЕЗУЛЬТАТЫ:")
    print(f"   Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"   Test RMSE:  {metrics['test_rmse']:.4f}")
    print(f"   Train R²:   {metrics['train_r2']:.4f}")
    print(f"   Test R²:    {metrics['test_r2']:.4f}")
    print(f"   CV RMSE:    {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
    
    # Переобучение
    overfit = metrics['train_rmse'] - metrics['test_rmse']
    print(f"   Переобучение: {overfit:.4f} ({'🟢 Низкое' if abs(overfit) < 0.1 else '🟡 Среднее' if abs(overfit) < 0.3 else '🔴 Высокое'})")
    
    return model, metrics

def compare_metrics(metrics1, metrics2, name1, name2):
    """Сравнивает метрики двух моделей"""
    print(f"\n🆚 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    
    key_metrics = ['test_rmse', 'test_r2', 'cv_rmse_mean']
    
    print(f"{'Метрика':<15} {name1:<15} {name2:<15} {'Улучшение':<15}")
    print("-" * 65)
    
    improvements = {}
    
    for metric in key_metrics:
        val1 = metrics1[metric]
        val2 = metrics2[metric]
        
        if metric in ['test_rmse', 'cv_rmse_mean']:
            # Для RMSE - меньше лучше
            improvement = ((val1 - val2) / val1) * 100
            improvement_text = f"{improvement:+.2f}%"
            is_better = val2 < val1
        else:
            # Для R² - больше лучше
            improvement = ((val2 - val1) / abs(val1)) * 100
            improvement_text = f"{improvement:+.2f}%"
            is_better = val2 > val1
        
        improvements[metric] = improvement
        
        status = "🟢" if is_better else "🔴"
        print(f"{metric:<15} {val1:<15.4f} {val2:<15.4f} {status} {improvement_text:<10}")
    
    return improvements

def generate_recommendation(dataset_stats1, dataset_stats2, improvements, name1, name2):
    """Генерирует рекомендацию по замене датасета"""
    print(f"\n🎯 РЕКОМЕНДАЦИЯ ПО ЗАМЕНЕ ДАТАСЕТА")
    print("=" * 50)
    
    # Анализ данных
    data_quality_score = 0
    
    # Размер датасета
    size_ratio = dataset_stats2['shape'][0] / dataset_stats1['shape'][0]
    if size_ratio > 1.1:
        print(f"✅ Размер данных: увеличение на {(size_ratio-1)*100:.1f}% ({dataset_stats2['shape'][0]:,} vs {dataset_stats1['shape'][0]:,})")
        data_quality_score += 1
    
    # Качество данных (меньше пропусков)
    if dataset_stats2['missing_total'] < dataset_stats1['missing_total']:
        print(f"✅ Качество данных: меньше пропусков ({dataset_stats2['missing_total']} vs {dataset_stats1['missing_total']})")
        data_quality_score += 1
    
    # Анализ метрик
    metric_improvements = 0
    key_metrics = ['test_rmse', 'test_r2', 'cv_rmse_mean']
    
    for metric in key_metrics:
        if metric in ['test_rmse', 'cv_rmse_mean']:
            if improvements[metric] > 0:  # Уменьшение RMSE - хорошо
                metric_improvements += 1
        else:
            if improvements[metric] > 0:  # Увеличение R² - хорошо
                metric_improvements += 1
    
    print(f"\n📊 ОЦЕНКА УЛУЧШЕНИЙ:")
    print(f"   Качество данных: {data_quality_score}/2")
    print(f"   Улучшение метрик: {metric_improvements}/{len(key_metrics)}")
    
    # Финальная рекомендация
    total_score = data_quality_score + metric_improvements
    max_score = 2 + len(key_metrics)
    
    if total_score >= max_score * 0.7:
        recommendation = "🟢 РЕКОМЕНДУЕТСЯ ЗАМЕНА"
        reason = "Улучшенный датасет показывает значительно лучшие результаты"
    elif total_score >= max_score * 0.4:
        recommendation = "🟡 РАССМОТРЕТЬ ЗАМЕНУ"
        reason = "Есть улучшения, но они не критичные"
    else:
        recommendation = "🔴 НЕ РЕКОМЕНДУЕТСЯ ЗАМЕНА"
        reason = "Улучшенный датасет не показывает значительных преимуществ"
    
    print(f"\n{recommendation}")
    print(f"Обоснование: {reason}")
    print(f"Общий балл: {total_score}/{max_score}")
    
    return recommendation, total_score, max_score

def main():
    """Основная функция сравнения"""
    print("🔍 АНАЛИЗ ВЛИЯНИЯ ЗАМЕНЫ ДАТАСЕТА НА МОДЕЛЬ")
    print("=" * 60)
    
    # Загружаем конфигурацию
    config = load_config()
    target_col = config['preprocessing']['features']['target_column']
    
    # Пути к датасетам
    dataset1_path = "data/initial_data_set.csv"
    dataset2_path = "data/merged_data_improved.csv"
    
    # Загружаем датасеты
    print("📥 Загрузка датасетов...")
    df1 = pd.read_csv(dataset1_path)
    df2 = pd.read_csv(dataset2_path)
    
    # Анализируем структуру данных
    stats1 = analyze_dataset(df1, "ИСХОДНЫЕ ДАННЫЕ")
    stats2 = analyze_dataset(df2, "УЛУЧШЕННЫЕ ДАННЫЕ")
    
    # Проверяем наличие целевой переменной
    if target_col not in df1.columns:
        raise ValueError(f"Целевая переменная '{target_col}' не найдена в исходных данных")
    if target_col not in df2.columns:
        raise ValueError(f"Целевая переменная '{target_col}' не найдена в улучшенных данных")
    
    # Подготавливаем данные для ML
    print(f"\n🔧 ПОДГОТОВКА ДАННЫХ ДЛЯ ML (target: {target_col})")
    X1, y1 = prepare_data_for_ml(df1, config, target_col)
    X2, y2 = prepare_data_for_ml(df2, config, target_col)
    
    # Обучаем модели и получаем метрики
    model1, metrics1 = train_and_evaluate_model(X1, y1, "ИСХОДНЫЕ ДАННЫЕ", config)
    model2, metrics2 = train_and_evaluate_model(X2, y2, "УЛУЧШЕННЫЕ ДАННЫЕ", config)
    
    # Сравниваем результаты
    improvements = compare_metrics(metrics1, metrics2, "Исходные", "Улучшенные")
    
    # Генерируем рекомендацию
    recommendation, score, max_score = generate_recommendation(
        stats1, stats2, improvements, "исходных", "улучшенных"
    )
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'timestamp': timestamp,
        'datasets': {
            'original': {
                'path': dataset1_path,
                'stats': stats1
            },
            'improved': {
                'path': dataset2_path,
                'stats': stats2
            }
        },
        'metrics': {
            'original': metrics1,
            'improved': metrics2,
            'improvements': improvements
        },
        'recommendation': {
            'decision': recommendation,
            'score': f"{score}/{max_score}",
            'timestamp': timestamp
        }
    }
    
    results_path = f"cv_results/dataset_comparison_{timestamp}.json"
    os.makedirs("cv_results", exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены: {results_path}")
    
    return recommendation, results

if __name__ == "__main__":
    try:
        recommendation, results = main()
        print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН: {recommendation}")
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        raise
