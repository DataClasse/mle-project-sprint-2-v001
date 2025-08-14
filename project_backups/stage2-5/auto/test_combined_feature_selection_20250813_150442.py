#!/usr/bin/env python3
"""
Тестирование нового комбинированного подхода Feature Selection
"""

# Подавление предупреждений
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
import yaml
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, r2_score

def load_config():
    """Загрузка конфигурации"""
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_and_prepare_data():
    """Загрузка и базовая подготовка данных"""
    print("📊 Загрузка данных...")
    
    # Загружаем конфигурацию для получения drop_columns
    config = load_config()
    
    # Загружаем данные
    df = pd.read_csv('data/initial_data_set.csv')
    print(f"✅ Загружено {df.shape[0]} записей, {df.shape[1]} столбцов")
    
    # Удаляем колонки согласно конфигурации
    drop_columns = config['preprocessing']['features']['drop_columns']
    print(f"🗑️ Удаляем колонки согласно конфигу: {drop_columns}")
    
    columns_to_drop = [col for col in drop_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
        print(f"✅ Удалено колонок: {columns_to_drop}")
    
    # Удаляем дополнительные служебные колонки
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Разделяем на признаки и целевую переменную
    target_col = config['preprocessing']['features']['target_column']
    features = df.drop(target_col, axis=1)
    target = df[target_col]
    
    print(f"🎯 Целевая переменная: {target_col}")
    print(f"📝 Признаков: {features.shape[1]}")
    print(f"📋 Список признаков: {list(features.columns)}")
    
    return features, target

def create_manual_features(df):
    """Создание ручных признаков как в Stage 3"""
    print("🔧 Создание ручных признаков...")
    
    df = df.copy()
    
    # Соотношения площадей
    df['kitchen_to_total_ratio'] = df['kitchen_area'] / df['total_area']
    df['living_to_total_ratio'] = df['living_area'] / df['total_area']
    
    # Площадь на комнату
    df['area_per_room'] = df['total_area'] / df['rooms']
    
    # Соотношение этажей
    df['floor_ratio'] = df['floor'] / df['floors_total']
    
    # Квадраты ключевых признаков
    df['total_area_sq'] = df['total_area'] ** 2
    
    # Расстояние от центра (примерное для тестирования)
    # Координаты центра Москвы: ~55.7558, 37.6176
    moscow_center_lat = 55.7558
    moscow_center_lon = 37.6176
    
    df['distance_from_center'] = np.sqrt(
        (df['latitude'] - moscow_center_lat)**2 + 
        (df['longitude'] - moscow_center_lon)**2
    ) * 111  # примерное расстояние в км
    
    print(f"✅ Создано {len(df.columns)} признаков (включая новые)")
    return df

def preprocess_features(features):
    """Предобработка признаков"""
    print("⚙️ Предобработка признаков...")
    
    features_processed = features.copy()
    
    # Кодирование категориальных переменных
    categorical_features = ['building_type']
    
    for col in categorical_features:
        if col in features_processed.columns:
            le = LabelEncoder()
            features_processed[col] = le.fit_transform(features_processed[col].astype(str))
    
    # Заполнение пропусков медианой для числовых признаков
    numeric_features = features_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_features:
        if features_processed[col].isnull().sum() > 0:
            median_val = features_processed[col].median()
            features_processed[col].fillna(median_val, inplace=True)
    
    print(f"✅ Предобработка завершена")
    return features_processed

def test_combined_feature_selection(X_train, X_test, y_train, y_test):
    """Тестирование комбинированного подхода Feature Selection"""
    print("\n🔬 ТЕСТИРОВАНИЕ КОМБИНИРОВАННОГО ПОДХОДА")
    print("=" * 60)
    
    start_time = time.time()
    
    # Этап 1: Feature Importance
    print("📊 Этап 1: Feature Importance для предварительной фильтрации")
    
    base_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )
    
    print("⏳ Анализ важности признаков...")
    base_model.fit(X_train, y_train)
    
    # Получаем важность признаков
    feature_importance = base_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Отбираем топ-20 признаков
    top_20_features = importance_df.head(20)['feature'].tolist()
    print(f"✅ Отобрано {len(top_20_features)} лучших признаков по важности")
    
    print("🔝 Топ-10 признаков по важности:")
    for i, row in importance_df.head(10).iterrows():
        print(f"   {row.name+1:2d}. {row['feature']:25s} {row['importance']:.4f}")
    
    # Этап 2: SFS внутри топ-20
    print(f"\n📊 Этап 2: Sequential Feature Selection внутри топ-{len(top_20_features)}")
    print("⚙️ Целевое количество признаков: 16 (по результатам экспериментов)")
    
    X_train_top20 = X_train[top_20_features]
    X_test_top20 = X_test[top_20_features]
    
    # SFS модель
    sfs_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )
    
    sfs = SFS(
        sfs_model,
        k_features=16,
        forward=True,
        floating=False,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=1
    )
    
    print("⏳ Запуск Sequential Feature Selection...")
    sfs_start = time.time()
    sfs = sfs.fit(X_train_top20, y_train)
    sfs_time = time.time() - sfs_start
    
    # Получаем финальный список выбранных признаков
    selected_features = list(sfs.k_feature_names_)
    
    print(f"✅ SFS завершен за {sfs_time:.1f} секунд")
    print(f"📊 Выбрано признаков: {len(selected_features)}")
    print(f"🎯 Выбранные признаки:")
    for i, feature in enumerate(selected_features, 1):
        print(f"   {i:2d}. {feature}")
    
    # Тестирование качества модели с выбранными признаками
    print(f"\n📈 Тестирование качества модели...")
    
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Обучение на выбранных признаках
    final_model.fit(X_train_selected, y_train)
    
    # Предсказания
    y_pred_train = final_model.predict(X_train_selected)
    y_pred_test = final_model.predict(X_test_selected)
    
    # Метрики
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(final_model, X_train_selected, y_train, 
                               cv=3, scoring='neg_root_mean_squared_error', n_jobs=1)
    cv_rmse = -cv_scores.mean()
    
    total_time = time.time() - start_time
    
    print(f"\n🎯 РЕЗУЛЬТАТЫ КОМБИНИРОВАННОГО ПОДХОДА:")
    print(f"   ⏱️  Общее время: {total_time:.1f} секунд")
    print(f"   📊 Признаков: {len(selected_features)} (из {X_train.shape[1]})")
    print(f"   🎯 Train RMSE: {train_rmse:,.0f} руб.")
    print(f"   🎯 Test RMSE:  {test_rmse:,.0f} руб.")
    print(f"   📈 Train R²:   {train_r2:.4f}")
    print(f"   📈 Test R²:    {test_r2:.4f}")
    print(f"   🔄 CV RMSE:    {cv_rmse:,.0f} руб.")
    
    return {
        'selected_features': selected_features,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'cv_rmse': cv_rmse,
        'execution_time': total_time,
        'n_features': len(selected_features)
    }

def test_baseline_comparison(X_train, X_test, y_train, y_test):
    """Сравнение с baseline (все признаки)"""
    print("\n📊 СРАВНЕНИЕ С BASELINE (все признаки)")
    print("=" * 50)
    
    baseline_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )
    
    start_time = time.time()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_time = time.time() - start_time
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    
    cv_scores_baseline = cross_val_score(baseline_model, X_train, y_train, 
                                        cv=3, scoring='neg_root_mean_squared_error', n_jobs=1)
    cv_rmse_baseline = -cv_scores_baseline.mean()
    
    print(f"🎯 BASELINE РЕЗУЛЬТАТЫ:")
    print(f"   ⏱️  Время обучения: {baseline_time:.1f} секунд")
    print(f"   📊 Признаков: {X_train.shape[1]}")
    print(f"   🎯 Test RMSE: {baseline_rmse:,.0f} руб.")
    print(f"   📈 Test R²:   {baseline_r2:.4f}")
    print(f"   🔄 CV RMSE:   {cv_rmse_baseline:,.0f} руб.")
    
    return {
        'test_rmse': baseline_rmse,
        'test_r2': baseline_r2,
        'cv_rmse': cv_rmse_baseline,
        'execution_time': baseline_time,
        'n_features': X_train.shape[1]
    }

def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ КОМБИНИРОВАННОГО FEATURE SELECTION")
    print("=" * 80)
    
    try:
        # Загрузка конфигурации
        config = load_config()
        print("✅ Конфигурация загружена")
        
        # Загрузка и подготовка данных
        features, target = load_and_prepare_data()
        
        # Создание ручных признаков
        features_enhanced = create_manual_features(features)
        
        # Предобработка
        features_processed = preprocess_features(features_enhanced)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features_processed, target, 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"\n📊 Разделение данных:")
        print(f"   🎯 Train: {X_train.shape[0]} образцов")
        print(f"   🎯 Test:  {X_test.shape[0]} образцов")
        print(f"   📝 Признаков: {X_train.shape[1]}")
        
        # Тестирование baseline
        baseline_results = test_baseline_comparison(X_train, X_test, y_train, y_test)
        
        # Тестирование комбинированного подхода
        combined_results = test_combined_feature_selection(X_train, X_test, y_train, y_test)
        
        # Сравнение результатов
        print(f"\n🏆 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        rmse_improvement = (baseline_results['test_rmse'] - combined_results['test_rmse']) / baseline_results['test_rmse'] * 100
        r2_improvement = (combined_results['test_r2'] - baseline_results['test_r2']) / baseline_results['test_r2'] * 100
        feature_reduction = (baseline_results['n_features'] - combined_results['n_features']) / baseline_results['n_features'] * 100
        
        print(f"📊 BASELINE:           RMSE={baseline_results['test_rmse']:,.0f}, R²={baseline_results['test_r2']:.4f}, Признаков={baseline_results['n_features']}")
        print(f"🔬 КОМБИНИРОВАННЫЙ:    RMSE={combined_results['test_rmse']:,.0f}, R²={combined_results['test_r2']:.4f}, Признаков={combined_results['n_features']}")
        print(f"")
        print(f"📈 Улучшение RMSE:     {rmse_improvement:+.2f}% ({'✅ УЛУЧШЕНИЕ' if rmse_improvement > 0 else '❌ УХУДШЕНИЕ'})")
        print(f"📈 Улучшение R²:       {r2_improvement:+.2f}% ({'✅ УЛУЧШЕНИЕ' if r2_improvement > 0 else '❌ УХУДШЕНИЕ'})")
        print(f"📉 Сокращение признаков: {feature_reduction:.1f}% ({'✅ СОКРАЩЕНИЕ' if feature_reduction > 0 else '⚠️ УВЕЛИЧЕНИЕ'})")
        
        # Проверка соответствия ожиданиям из экспериментов
        print(f"\n🎯 ПРОВЕРКА ОЖИДАНИЙ ИЗ ЭКСПЕРИМЕНТОВ:")
        if rmse_improvement > 0.3:  # Ожидали +0.53%
            print(f"✅ RMSE улучшение соответствует ожиданиям ({rmse_improvement:.2f}% > 0.3%)")
        else:
            print(f"⚠️ RMSE улучшение ниже ожиданий ({rmse_improvement:.2f}% < 0.3%)")
            
        if combined_results['n_features'] == 16:
            print(f"✅ Количество признаков соответствует ожиданиям (16)")
        else:
            print(f"⚠️ Количество признаков отличается от ожиданий ({combined_results['n_features']} != 16)")
        
        print(f"\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
