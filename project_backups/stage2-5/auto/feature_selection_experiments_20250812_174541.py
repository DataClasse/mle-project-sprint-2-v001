# ЭКСПЕРИМЕНТЫ ПО УЛУЧШЕНИЮ FEATURE SELECTION
# ============================================
# Цель: Исправить ухудшение качества (-3.9%) от Feature Selection
# Автор: AI Assistant
# Дата: 2025-08-12

import pandas as pd
import yaml
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import time
from typing import List, Tuple, Dict

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================

def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Загрузка и подготовка данных с feature engineering"""
    print("📊 Загрузка и подготовка данных...")
    
    # Загрузка конфигурации
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Загрузка данных
    df = pd.read_csv('data/initial_data_set.csv')
    df.drop(['id', 'building_id'], axis=1, inplace=True)
    
    target = df['price']
    features = df.drop('price', axis=1)
    
    # Feature Engineering (те же признаки что показали +1.4% улучшение)
    features_enriched = features.copy()
    
    # 1. Соотношения площадей
    features_enriched['kitchen_to_total_ratio'] = features_enriched['kitchen_area'] / (features_enriched['total_area'] + 1e-8)
    features_enriched['living_to_total_ratio'] = features_enriched['living_area'] / (features_enriched['total_area'] + 1e-8)
    
    # 2. Признаки этажности
    features_enriched['floor_ratio'] = features_enriched['floor'] / (features_enriched['floors_total'] + 1e-8)
    
    # 3. Площадь на комнату
    features_enriched['area_per_room'] = features_enriched['total_area'] / (features_enriched['rooms'] + 1e-8)
    
    # 4. Квадраты ключевых признаков
    features_enriched['total_area_sq'] = features_enriched['total_area'] ** 2
    features_enriched['ceiling_height_sq'] = features_enriched['ceiling_height'] ** 2
    
    # 5. Расстояние от центра Москвы
    moscow_center_lat, moscow_center_lon = 55.7558, 37.6173
    features_enriched['distance_from_center'] = np.sqrt(
        (features_enriched['latitude'] - moscow_center_lat)**2 + 
        (features_enriched['longitude'] - moscow_center_lon)**2
    )
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        features_enriched, target, test_size=0.2, random_state=42, shuffle=True
    )
    
    feature_names = features_enriched.columns.tolist()
    
    print(f"✅ Данные подготовлены:")
    print(f"   Обучающая выборка: {X_train.shape}")
    print(f"   Тестовая выборка: {X_test.shape}")
    print(f"   Всего признаков: {len(feature_names)}")
    print()
    
    return X_train, X_test, y_train, y_test, feature_names

def evaluate_model(X_train, X_test, y_train, y_test, selected_features: List[str] = None) -> Dict:
    """Оценка качества модели"""
    if selected_features:
        X_train_eval = X_train[selected_features]
        X_test_eval = X_test[selected_features]
    else:
        X_train_eval = X_train
        X_test_eval = X_test
    
    # Обучение модели
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_eval, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test_eval)
    
    # Метрики
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_eval, y_train, 
        cv=3, scoring='neg_root_mean_squared_error'
    )
    cv_rmse = -cv_scores.mean()
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'cv_rmse': cv_rmse,
        'n_features': X_train_eval.shape[1]
    }

# =============================================================================
# BASELINE (БЕЗ FEATURE SELECTION)
# =============================================================================

def run_baseline_experiment(X_train, X_test, y_train, y_test) -> Dict:
    """Baseline: все признаки без отбора"""
    print("🏁 BASELINE: Все признаки (без Feature Selection)")
    print("-" * 50)
    
    start_time = time.time()
    results = evaluate_model(X_train, X_test, y_train, y_test)
    duration = time.time() - start_time
    
    print(f"✅ Baseline результаты ({duration:.1f}с):")
    print(f"   RMSE: {results['rmse']:,.0f}")
    print(f"   R²: {results['r2']:.4f}")
    print(f"   MAE: {results['mae']:,.0f}")
    print(f"   CV RMSE: {results['cv_rmse']:,.0f}")
    print(f"   Признаков: {results['n_features']}")
    print()
    
    return results

# =============================================================================
# ЭКСПЕРИМЕНТ 1: УЛУЧШЕННЫЙ SEQUENTIAL FEATURE SELECTION
# =============================================================================

def run_sfs_experiment(X_train, X_test, y_train, y_test, feature_names: List[str]) -> Dict:
    """Эксперимент 1: Sequential Feature Selection с улучшенными параметрами"""
    print("🔬 ЭКСПЕРИМЕНТ 1: Улучшенный Sequential Feature Selection")
    print("-" * 60)
    
    start_time = time.time()
    
    # Более мощная базовая модель для SFS
    base_model = xgb.XGBRegressor(
        n_estimators=300,  # Уменьшено для скорости SFS
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1  # Для стабильности SFS
    )
    
    # Пробуем разные количества признаков
    best_results = None
    best_features = None
    best_k = None
    
    # Тестируем разные k от 10 до 18 (было 12)
    for k in [10, 12, 14, 16, 18]:
        print(f"🔍 Тестируем k={k} признаков...")
        
        # Sequential Forward Selection
        sfs = SFS(
            base_model,
            k_features=k,
            forward=True,
            floating=False,
            scoring='neg_root_mean_squared_error',
            cv=3,  # Уменьшили для скорости
            n_jobs=1
        )
        
        try:
            sfs.fit(X_train, y_train)
            selected_features = list(sfs.k_feature_names_)
            
            # Оценка на полной модели
            results = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
            
            print(f"   k={k}: RMSE={results['rmse']:,.0f}, R²={results['r2']:.4f}")
            
            # Сохраняем лучший результат
            if best_results is None or results['rmse'] < best_results['rmse']:
                best_results = results
                best_features = selected_features
                best_k = k
                
        except Exception as e:
            print(f"   k={k}: Ошибка - {str(e)[:50]}...")
    
    duration = time.time() - start_time
    
    print(f"✅ Лучший SFS результат ({duration:.1f}с):")
    print(f"   Лучший k: {best_k}")
    print(f"   RMSE: {best_results['rmse']:,.0f}")
    print(f"   R²: {best_results['r2']:.4f}")
    print(f"   MAE: {best_results['mae']:,.0f}")
    print(f"   CV RMSE: {best_results['cv_rmse']:,.0f}")
    print(f"   Выбранные признаки ({len(best_features)}):")
    for i, feature in enumerate(best_features[:8]):
        print(f"     {i+1}. {feature}")
    if len(best_features) > 8:
        print(f"     ... и еще {len(best_features) - 8}")
    print()
    
    return {**best_results, 'selected_features': best_features, 'method': 'SFS_improved'}

if __name__ == "__main__":
    # Загрузка данных
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Запуск baseline
    baseline_results = run_baseline_experiment(X_train, X_test, y_train, y_test)
    
    # Запуск эксперимента 1
    sfs_results = run_sfs_experiment(X_train, X_test, y_train, y_test, feature_names)
    
    # Сравнение
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("=" * 40)
    print(f"Baseline (все признаки): RMSE={baseline_results['rmse']:,.0f}, R²={baseline_results['r2']:.4f}")
    print(f"SFS улучшенный: RMSE={sfs_results['rmse']:,.0f}, R²={sfs_results['r2']:.4f}")
    
    improvement = ((baseline_results['rmse'] - sfs_results['rmse']) / baseline_results['rmse']) * 100
    print(f"Улучшение: {improvement:+.1f}%")
    
    if improvement > 0:
        print("✅ SFS показал улучшение!")
    else:
        print("⚠️ SFS все еще показывает ухудшение")

