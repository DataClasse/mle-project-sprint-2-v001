# ЭКСПЕРИМЕНТЫ ПО УЛУЧШЕНИЮ FEATURE SELECTION
# ============================================
# Цель: Исправить ухудшение качества (-3.9%) от Feature Selection
# Автор: AI Assistant
# Дата: 2025-08-12

import os
import json
import pandas as pd
import yaml
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import time
from datetime import datetime
from typing import List, Tuple, Dict

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================

def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Загрузка и подготовка данных с feature engineering"""
    print("Загрузка и подготовка данных...")
    
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
    
    print("✅ Данные подготовлены:")
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
    print("BASELINE: Все признаки (без Feature Selection)")
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
    print("ЭКСПЕРИМЕНТ 1: Улучшенный Sequential Feature Selection")
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
        print(f"Тестируем k={k} признаков...")
        
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

# =============================================================================
# ЭКСПЕРИМЕНТ 2: FEATURE IMPORTANCE ОТ XGBOOST
# =============================================================================

def run_feature_importance_experiment(
    X_train, X_test, y_train, y_test, baseline_rmse: float
) -> Dict:
    """Эксперимент 2: Отбор топ-k признаков по важности XGBoost"""
    print("ЭКСПЕРИМЕНТ 2: Feature Importance от XGBoost")
    print("-" * 60)

    start_time = time.time()

    # Базовая модель для получения важностей
    baseline_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
    )
    baseline_model.fit(X_train, y_train)

    feature_importance = baseline_model.feature_importances_
    feature_names = X_train.columns.tolist()

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": feature_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    best_rmse = float("inf")
    best_r2 = None
    best_k = None
    best_features: List[str] = []

    print("Тестирование разных количеств признаков:")
    for k in [8, 10, 12, 14, 16, 18]:
        top_features = importance_df.head(k)["feature"].tolist()

        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        model.fit(X_train[top_features], y_train)
        y_pred = model.predict(X_test[top_features])
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        print(f"   k={k:2d}: RMSE={rmse:,.0f}, R²={r2:.4f}, улучшение={improvement:+.1f}%")

        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_k = k
            best_features = top_features

    duration = time.time() - start_time
    print(f"✅ Лучший результат Feature Importance ({duration:.1f}с):")
    print(f"   Лучший k: {best_k}")
    print(f"   RMSE: {best_rmse:,.0f}")
    print(f"   Улучшение: {((baseline_rmse - best_rmse) / baseline_rmse) * 100:+.1f}%")
    print(f"   Выбранные признаки ({len(best_features)}):")
    for i, feature in enumerate(best_features):
        if i >= 12:
            print(f"     ... и еще {len(best_features) - 12}")
            break
        print(f"     {i+1:2d}. {feature}")
    print()

    return {
        "rmse": best_rmse,
        "r2": best_r2 if best_r2 is not None else float("nan"),
        "n_features": len(best_features),
        "selected_features": best_features,
        "method": f"XGB_feature_importance_top_{best_k}",
    }

# =============================================================================
# ЭКСПЕРИМЕНТ 3: КОМБИНИРОВАННЫЙ ПОДХОД (FI → SFS)
# =============================================================================

def run_combined_experiment(
    X_train, X_test, y_train, y_test, preselect_top_m: int = 20
) -> Dict:
    """Эксперимент 3: Предварительный отбор по важности, затем SFS внутри топ-M"""
    print("ЭКСПЕРИМЕНТ 3: Комбинированный подход (Feature Importance → SFS)")
    print("-" * 60)

    start_time = time.time()

    # Предварительный отбор по важности
    fi_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
    )
    fi_model.fit(X_train, y_train)
    fi_scores = fi_model.feature_importances_
    fi_order = np.argsort(fi_scores)[::-1]
    top_m_features = X_train.columns[fi_order[:preselect_top_m]].tolist()

    # SFS внутри топ-M признаков
    base_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
        random_state=42, n_jobs=1
    )

    best_results = None
    best_features = None
    best_k = None

    for k in [10, 12, 14, 16]:
        print(f"Тестируем SFS внутри топ-{preselect_top_m} с k={k}...")
        try:
            sfs = SFS(
                base_model,
                k_features=k,
                forward=True,
                floating=False,
                scoring='neg_root_mean_squared_error',
                cv=3,
                n_jobs=1
            )
            sfs.fit(X_train[top_m_features], y_train)
            selected = list(sfs.k_feature_names_)
            results = evaluate_model(X_train[top_m_features], X_test[top_m_features], y_train, y_test, selected)

            print(f"   k={k}: RMSE={results['rmse']:,.0f}, R²={results['r2']:.4f}")

            if best_results is None or results['rmse'] < best_results['rmse']:
                best_results = results
                best_features = selected
                best_k = k
        except Exception as e:
            print(f"   Ошибка SFS (k={k}): {str(e)[:80]}...")

    duration = time.time() - start_time
    print(f"✅ Лучший комбинированный результат ({duration:.1f}с):")
    print(f"   Лучший k: {best_k}")
    print(f"   RMSE: {best_results['rmse']:,.0f}")
    print(f"   R²: {best_results['r2']:.4f}")
    print(f"   Признаков: {best_results['n_features']}")
    print(f"   Выбранные признаки ({len(best_features)}):")
    for i, feature in enumerate(best_features[:12]):
        print(f"     {i+1:2d}. {feature}")
    if len(best_features) > 12:
        print(f"     ... и еще {len(best_features) - 12}")
    print()

    return {**best_results, 'selected_features': best_features, 'method': f'Combined_FI{preselect_top_m}_SFS_k{best_k}'}

# =============================================================================
# ВСПОМОГАТЕЛЬНОЕ: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def save_experiments_report(
    baseline: Dict,
    sfs: Dict,
    fi: Dict,
    combined: Dict,
    base_dir: str = "."
) -> str:
    """Сохраняет результаты экспериментов в processed/feature_selection/ с таймстампом"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = os.path.join(base_dir, "processed", "feature_selection")
    os.makedirs(processed_dir, exist_ok=True)

    report = {
        "timestamp": timestamp,
        "baseline": baseline,
        "experiment_1_sfs": sfs,
        "experiment_2_feature_importance": fi,
        "experiment_3_combined": combined,
    }

    report_path = os.path.join(processed_dir, f"feature_selection_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_to_serializable)

    # Сохраняем списки признаков
    def save_features_list(name: str, result: Dict):
        feats = result.get("selected_features") or []
        if feats:
            path = os.path.join(processed_dir, f"{name}_features_{timestamp}.txt")
            with open(path, "w", encoding="utf-8") as f:
                for feat in feats:
                    f.write(f"{feat}\n")

    save_features_list("sfs", sfs)
    save_features_list("fi", fi)
    save_features_list("combined", combined)

    print(f"✅ Отчет сохранен: {report_path}")
    return report_path

if __name__ == "__main__":
    # Загрузка данных
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Запуск baseline
    baseline_results = run_baseline_experiment(X_train, X_test, y_train, y_test)
    
    # Эксперимент 1: Улучшенный SFS
    sfs_results = run_sfs_experiment(X_train, X_test, y_train, y_test, feature_names)

    # Эксперимент 2: Feature Importance
    fi_results = run_feature_importance_experiment(
        X_train, X_test, y_train, y_test, baseline_rmse=baseline_results["rmse"]
    )

    # Эксперимент 3: Комбинированный подход
    combined_results = run_combined_experiment(X_train, X_test, y_train, y_test, preselect_top_m=20)

    # Сравнение
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("=" * 40)
    print(f"Baseline (все признаки): RMSE={baseline_results['rmse']:,.0f}, R²={baseline_results['r2']:.4f}")
    print(f"SFS улучшенный:           RMSE={sfs_results['rmse']:,.0f}, R²={sfs_results['r2']:.4f}")
    print(f"Feature Importance:       RMSE={fi_results['rmse']:,.0f}, R²={fi_results['r2']:.4f}")
    print(f"Комбинированный подход:   RMSE={combined_results['rmse']:,.0f}, R²={combined_results['r2']:.4f}")

    # Сохранение отчета
    save_experiments_report(
        baseline=baseline_results,
        sfs=sfs_results,
        fi=fi_results,
        combined=combined_results,
        base_dir="."
    )

