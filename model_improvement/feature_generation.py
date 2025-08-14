#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Этап 3: Генерация признаков и обучение модели
Проект: Улучшение baseline-модели для Яндекс Недвижимости

Требования проекта:
1. Займитесь предобработкой данных. Используйте подходящие методы из sklearn.preprocessing
2. Сгенерируйте признаки, используя sklearn.preprocessing. Примените как минимум два метода:
   - PolynomialFeatures — для создания полиномиальных признаков
   - KBinsDiscretizer — для дискретизации числовых признаков
3. Соберите все преобразования в объект ColumnTransformer
4. Создайте sklearn-пайплайн. Интегрируйте ваш ColumnTransformer в объект Pipeline
5. Настройте автоматическую генерацию признаков с помощью библиотеки autofeat
6. Обучите новую версию модели на обогащённом наборе признаков
7. Оцените её качество и производительность
8. Результаты залогируйте в MLflow
9. Сохраните новую версию модели в MLflow Model Registry
"""

import os
import warnings
import time
import yaml
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Настройки для корректного отображения русского текста
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
warnings.filterwarnings("ignore")

class FeatureGenerator:
    """Класс для генерации признаков и обучения модели"""
    
    def __init__(self, data_path, config_path="../config.yaml"):
        """
        Инициализация генератора признаков
        
        Args:
            data_path (str): Путь к файлу данных
            config_path (str): Путь к конфигурационному файлу
        """
        self.data_path = data_path
        self.config_path = config_path
        self.df = None
        self.config = None
        self.target_column = None
        self.drop_columns = None
        self.numerical_features = None
        self.categorical_features = None
        self.boolean_features = None
        
        # Загружаем конфигурацию
        self._load_config()
        
        # Настройки отображения
        self._setup_display()
        
        # Результаты
        self.results = {}
        
    def _load_config(self):
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.target_column = self.config['preprocessing']['features']['target_column']
            self.drop_columns = self.config['preprocessing']['features']['drop_columns']
            self.numerical_features = self.config['preprocessing']['features']['numerical_features']
            self.categorical_features = self.config['preprocessing']['features']['categorical_numeric_features']
            self.boolean_features = self.config['preprocessing']['features']['boolean_features']
            
            print("✅ Конфигурация загружена успешно")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            # Значения по умолчанию
            self.target_column = 'price'
            self.drop_columns = ['id', 'building_id']
            self.numerical_features = ['total_area', 'rooms', 'floor', 'build_year']
            self.categorical_features = ['building_type']
            self.boolean_features = ['is_apartment', 'studio', 'has_elevator']
    
    def _setup_display(self):
        """Настройка параметров отображения"""
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 64
        sns.set_theme(style="ticks", palette="pastel")
        plt.style.use('seaborn-v0_8')
    
    def load_data(self):
        """Загрузка данных с предварительной обработкой"""
        print("📊 ЗАГРУЗКА ДАННЫХ...")
        print("=" * 50)
        
        # Проверяем существование файла
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
        
        # Загружаем данные с оптимизацией типов
        try:
            self.df = pd.read_csv(self.data_path, dtype={
                "rooms": "category",
                "building_type": "category",
                "floor": "int16",
                "floors_total": "int16",
                "flats_count": "int32",
                "build_year": "int16",
                "is_apartment": "bool",
                "studio": "bool",
                "has_elevator": "bool"
            })
            
            print(f"✅ Данные загружены успешно")
            print(f"   Размер: {self.df.shape[0]:,} строк × {self.df.shape[1]} столбцов")
            print(f"   Использование памяти: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке данных: {e}")
            raise
        
        # Удаляем ненужные столбцы
        columns_to_drop = list(set(self.df.columns) & set(self.drop_columns))
        if columns_to_drop:
            self.df.drop(columns=columns_to_drop, axis=1, inplace=True)
            print(f"   Удалены столбцы: {columns_to_drop}")
            print(f"   Новый размер: {self.df.shape[0]:,} строк × {self.df.shape[1]} столбцов")
        
        return self.df
    
    def manual_feature_engineering(self):
        """Ручная генерация признаков на основе доменной экспертизы"""
        print("\n🔧 РУЧНАЯ ГЕНЕРАЦИЯ ПРИЗНАКОВ")
        print("=" * 50)
        
        start_time = time.time()
        
        # Создаем копию данных для feature engineering
        df_features = self.df.copy()
        
        print("📊 Создаем новые признаки...")
        
        # 1. Площадь на комнату
        df_features['area_per_room'] = df_features['total_area'] / df_features['rooms'].astype(float)
        print("   ✅ area_per_room - площадь на комнату")
        
        # 2. Квадрат общей площади
        df_features['total_area_sq'] = df_features['total_area'] ** 2
        print("   ✅ total_area_sq - квадрат общей площади")
        
        # 3. Соотношение жилой и общей площади
        df_features['living_to_total_ratio'] = df_features['living_area'] / df_features['total_area']
        print("   ✅ living_to_total_ratio - соотношение жилой и общей площади")
        
        # 4. Соотношение кухонной и общей площади
        df_features['kitchen_to_total_ratio'] = df_features['kitchen_area'] / df_features['total_area']
        print("   ✅ kitchen_to_total_ratio - соотношение кухонной и общей площади")
        
        # 5. Возраст здания
        current_year = datetime.now().year
        df_features['building_age'] = current_year - df_features['build_year']
        print("   ✅ building_age - возраст здания")
        
        # 6. Плотность застройки (квартир на этаж)
        df_features['flats_per_floor'] = df_features['flats_count'] / df_features['floors_total']
        print("   ✅ flats_per_floor - плотность застройки")
        
        # 7. Расстояние от центра (если есть координаты)
        if 'latitude' in df_features.columns and 'longitude' in df_features.columns:
            # Центр Москвы (примерные координаты)
            moscow_center_lat, moscow_center_lon = 55.7558, 37.6176
            df_features['distance_from_center'] = np.sqrt(
                (df_features['latitude'] - moscow_center_lat) ** 2 + 
                (df_features['longitude'] - moscow_center_lon) ** 2
            ) * 111000  # Примерно в метрах
            print("   ✅ distance_from_center - расстояние от центра Москвы")
        
        # 8. Логарифм цены (для нормализации)
        df_features['log_price'] = np.log1p(df_features[self.target_column])
        print("   ✅ log_price - логарифм цены")
        
        # 9. Бинарные признаки для этажности
        df_features['is_high_floor'] = (df_features['floor'] > df_features['floors_total'] * 0.7).astype(int)
        df_features['is_low_floor'] = (df_features['floor'] <= 3).astype(int)
        print("   ✅ is_high_floor, is_low_floor - бинарные признаки этажности")
        
        # 10. Комбинированный признак типа здания и года постройки
        df_features['building_type_age'] = df_features['building_type'].astype(str) + '_' + \
                                         df_features['build_year'].astype(str)
        print("   ✅ building_type_age - комбинированный признак")
        
        # Обрабатываем бесконечные значения
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Заполняем пропущенные значения по типам данных
        for col in df_features.columns:
            if df_features[col].dtype.name == 'category':
                # Для категориальных признаков используем mode (самое частое значение)
                mode_value = df_features[col].mode().iloc[0] if not df_features[col].mode().empty else df_features[col].iloc[0]
                df_features[col] = df_features[col].fillna(mode_value)
            elif df_features[col].dtype.name == 'bool':
                # Для булевых признаков используем False
                df_features[col] = df_features[col].fillna(False)
            else:
                # Для числовых признаков используем 0
                df_features[col] = df_features[col].fillna(0)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Ручная генерация признаков завершена за {elapsed_time:.2f} сек")
        print(f"   Новый размер: {df_features.shape[0]:,} строк × {df_features.shape[1]} столбцов")
        
        self.df_features = df_features
        return df_features
    
    def create_sklearn_preprocessor(self):
        """Создание sklearn препроцессора с ColumnTransformer"""
        print("\n🔧 СОЗДАНИЕ SKLEARN ПРЕПРОЦЕССОРА")
        print("=" * 50)
        
        # Определяем числовые признаки (включая новые)
        numerical_features = [col for col in self.df_features.columns 
                            if col not in [self.target_column] + self.categorical_features + self.boolean_features]
        
        # Определяем категориальные признаки
        categorical_features = [col for col in self.categorical_features 
                              if col in self.df_features.columns]
        
        # Определяем булевы признаки
        boolean_features = [col for col in self.boolean_features 
                          if col in self.df_features.columns]
        
        print(f"📊 Числовые признаки: {len(numerical_features)}")
        print(f"📊 Категориальные признаки: {len(categorical_features)}")
        print(f"📊 Булевы признаки: {len(boolean_features)}")
        
        # Создаем трансформеры для разных типов признаков
        transformers = []
        
        # 1. Числовые признаки
        if numerical_features:
            numerical_transformer = Pipeline([
                ('scaler', StandardScaler()),
                ('polynomial', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
                ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'))
            ])
            transformers.append(('numerical', numerical_transformer, numerical_features))
            print("   ✅ Числовой трансформер создан")
        
        # 2. Категориальные признаки
        if categorical_features:
            # Используем OrdinalEncoder вместо LabelEncoder для совместимости с Pipeline
            from sklearn.preprocessing import OrdinalEncoder
            categorical_transformer = Pipeline([
                ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('categorical', categorical_transformer, categorical_features))
            print("   ✅ Категориальный трансформер создан")
        
        # 3. Булевы признаки (оставляем как есть)
        if boolean_features:
            transformers.append(('boolean', 'passthrough', boolean_features))
            print("   ✅ Булев трансформер создан")
        
        # Создаем ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            sparse_threshold=0.3
        )
        
        print(f"\n✅ ColumnTransformer создан с {len(transformers)} трансформерами")
        
        self.preprocessor = preprocessor
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.boolean_features = boolean_features
        
        return preprocessor
    
    def create_sklearn_pipeline(self):
        """Создание sklearn пайплайна"""
        print("\n🔧 СОЗДАНИЕ SKLEARN ПАЙПЛАЙНА")
        print("=" * 50)
        
        # Создаем XGBoost модель
        xgb_model = xgb.XGBRegressor(
            n_estimators=self.config['model']['params']['n_estimators'],
            max_depth=self.config['model']['params']['max_depth'],
            learning_rate=self.config['model']['params']['learning_rate'],
            subsample=self.config['model']['params']['subsample'],
            colsample_bytree=self.config['model']['params']['colsample_bytree'],
            random_state=self.config['train']['random_state'],
            eval_metric=self.config['model']['params']['eval_metric'],
            objective=self.config['model']['params']['objective'],
            tree_method=self.config['model']['params']['tree_method'],
            n_jobs=self.config['model']['params']['n_jobs']
        )
        
        # Создаем пайплайн
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', xgb_model)
        ])
        
        print("✅ Sklearn пайплайн создан:")
        print("   1. Preprocessor (ColumnTransformer)")
        print("   2. XGBoost Regressor")
        
        self.pipeline = pipeline
        return pipeline
    
    def prepare_data_for_training(self):
        """Подготовка данных для обучения"""
        print("\n🔧 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        print("=" * 50)
        
        # Разделяем признаки и целевую переменную
        X = self.df_features.drop(columns=[self.target_column])
        y = self.df_features[self.target_column]
        
        print(f"📊 Признаки: {X.shape[1]} столбцов")
        print(f"📊 Целевая переменная: {y.shape[0]} значений")
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['train']['test_size'],
            random_state=self.config['train']['random_state'],
            shuffle=self.config['train']['shuffle']
        )
        
        print(f"📊 Обучающая выборка: {X_train.shape[0]:,} строк")
        print(f"📊 Тестовая выборка: {X_test.shape[0]:,} строк")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_pipeline_model(self):
        """Обучение модели через пайплайн"""
        print("\n🚀 ОБУЧЕНИЕ МОДЕЛИ ЧЕРЕЗ ПАЙПЛАЙН")
        print("=" * 50)
        
        start_time = time.time()
        
        # Обучаем пайплайн
        print("📊 Обучение пайплайна...")
        self.pipeline.fit(self.X_train, self.y_train)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Обучение завершено за {elapsed_time:.2f} сек")
        
        # Предсказания
        y_train_pred = self.pipeline.predict(self.X_train)
        y_test_pred = self.pipeline.predict(self.X_test)
        
        # Сохраняем предсказания
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        
        return y_train_pred, y_test_pred
    
    def evaluate_model(self):
        """Оценка качества модели"""
        print("\n📊 ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
        print("=" * 50)
        
        # Метрики для обучающей выборки
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        
        # Метрики для тестовой выборки
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        # MAPE
        train_mape = np.mean(np.abs((self.y_train - self.y_train_pred) / self.y_train)) * 100
        test_mape = np.mean(np.abs((self.y_test - self.y_test_pred) / self.y_test)) * 100
        
        print("📊 МЕТРИКИ КАЧЕСТВА:")
        print("=" * 30)
        print(f"Обучающая выборка:")
        print(f"  RMSE: {train_rmse:,.0f}")
        print(f"  MAE:  {train_mae:,.0f}")
        print(f"  R²:   {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.2f}%")
        print()
        print(f"Тестовая выборка:")
        print(f"  RMSE: {test_rmse:,.0f}")
        print(f"  MAE:  {test_mae:,.0f}")
        print(f"  R²:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")
        
        # Сохраняем результаты
        self.results = {
            'train': {
                'rmse': train_rmse,
                'mae': train_mae,
                'r2': train_r2,
                'mape': train_mape
            },
            'test': {
                'rmse': test_rmse,
                'mae': test_mae,
                'r2': test_r2,
                'mape': test_mape
            }
        }
        
        return self.results
    
    def cross_validation_analysis(self):
        """Анализ кросс-валидации"""
        print("\n🔄 АНАЛИЗ КРОСС-ВАЛИДАЦИИ")
        print("=" * 50)
        
        # Кросс-валидация
        cv_scores = cross_val_score(
            self.pipeline,
            self.X_train,
            self.y_train,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        # Преобразуем в положительные значения
        cv_rmse = -cv_scores
        cv_mean = cv_rmse.mean()
        cv_std = cv_rmse.std()
        
        print(f"📊 Кросс-валидация (5 фолдов):")
        print(f"  Средний RMSE: {cv_mean:,.0f} ± {cv_std:,.0f}")
        print(f"  Отдельные фолды:")
        for i, score in enumerate(cv_rmse, 1):
            print(f"    Фолд {i}: {score:,.0f}")
        
        # Сохраняем результаты CV
        self.results['cross_validation'] = {
            'mean_rmse': cv_mean,
            'std_rmse': cv_std,
            'fold_scores': cv_rmse.tolist()
        }
        
        return cv_mean, cv_std
    
    def feature_importance_analysis(self):
        """Анализ важности признаков"""
        print("\n🔍 АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("=" * 50)
        
        # Получаем названия признаков после препроцессинга
        feature_names = []
        
        # Числовые признаки
        if hasattr(self.preprocessor, 'named_transformers_'):
            if 'numerical' in self.preprocessor.named_transformers_:
                numerical_transformer = self.preprocessor.named_transformers_['numerical']
                if hasattr(numerical_transformer, 'named_steps_'):
                    if 'polynomial' in numerical_transformer.named_steps_:
                        poly_transformer = numerical_transformer.named_steps_['polynomial']
                        feature_names.extend(poly_transformer.get_feature_names_out(self.numerical_features))
                    else:
                        feature_names.extend(self.numerical_features)
                else:
                    feature_names.extend(self.numerical_features)
            else:
                feature_names.extend(self.numerical_features)
        
        # Категориальные признаки
        feature_names.extend(self.categorical_features)
        
        # Булевы признаки
        feature_names.extend(self.boolean_features)
        
        # Получаем важность признаков из XGBoost
        if hasattr(self.pipeline, 'named_steps_'):
            xgb_model = self.pipeline.named_steps_['regressor']
            if hasattr(xgb_model, 'feature_importances_'):
                importances = xgb_model.feature_importances_
                
                # Создаем DataFrame с важностью признаков
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("🔝 Топ-15 важных признаков:")
                print(feature_importance_df.head(15))
                
                # Визуализация важности признаков
                self._plot_feature_importance(feature_importance_df)
                
                # Сохраняем результаты
                self.results['feature_importance'] = feature_importance_df.to_dict('records')
                
                return feature_importance_df
        
        print("❌ Не удалось получить важность признаков")
        return None
    
    def _plot_feature_importance(self, feature_importance_df):
        """Визуализация важности признаков"""
        # Создаем папку для графиков
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Топ-20 признаков
        top_features = feature_importance_df.head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title("Важность признаков (топ-20)")
        plt.xlabel("Важность")
        plt.ylabel("Признак")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_visualizations(self):
        """Создание визуализаций результатов"""
        print("\n🎨 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("=" * 50)
        
        # Создаем папку для графиков
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Actual vs Predicted
        self._plot_actual_vs_predicted(plots_dir)
        
        # 2. Residuals plot
        self._plot_residuals(plots_dir)
        
        # 3. Distribution of predictions
        self._plot_prediction_distribution(plots_dir)
        
        print(f"✅ Графики сохранены в папку: {plots_dir}")
    
    def _plot_actual_vs_predicted(self, plots_dir):
        """График фактических vs предсказанных значений"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Обучающая выборка
        axes[0].scatter(self.y_train, self.y_train_pred, alpha=0.6)
        axes[0].plot([self.y_train.min(), self.y_train.max()], 
                     [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0].set_xlabel('Фактические значения')
        axes[0].set_ylabel('Предсказанные значения')
        axes[0].set_title('Обучающая выборка')
        axes[0].grid(True, alpha=0.3)
        
        # Тестовая выборка
        axes[1].scatter(self.y_test, self.y_test_pred, alpha=0.6)
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Фактические значения')
        axes[1].set_ylabel('Предсказанные значения')
        axes[1].set_title('Тестовая выборка')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "actual_vs_predicted.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_residuals(self, plots_dir):
        """График остатков"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Остатки обучающей выборки
        train_residuals = self.y_train - self.y_train_pred
        axes[0].scatter(self.y_train_pred, train_residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Предсказанные значения')
        axes[0].set_ylabel('Остатки')
        axes[0].set_title('Остатки (обучающая выборка)')
        axes[0].grid(True, alpha=0.3)
        
        # Остатки тестовой выборки
        test_residuals = self.y_test - self.y_test_pred
        axes[1].scatter(self.y_test_pred, test_residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Предсказанные значения')
        axes[1].set_ylabel('Остатки')
        axes[1].set_title('Остатки (тестовая выборка)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "residuals.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_prediction_distribution(self, plots_dir):
        """Распределение предсказаний"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Распределение предсказаний обучающей выборки
        axes[0].hist(self.y_train_pred, bins=50, alpha=0.7, label='Предсказания')
        axes[0].hist(self.y_train, bins=50, alpha=0.7, label='Фактические')
        axes[0].set_xlabel('Цена')
        axes[0].set_ylabel('Частота')
        axes[0].set_title('Распределение (обучающая выборка)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Распределение предсказаний тестовой выборки
        axes[1].hist(self.y_test_pred, bins=50, alpha=0.7, label='Предсказания')
        axes[1].hist(self.y_test, bins=50, alpha=0.7, label='Фактические')
        axes[1].set_xlabel('Цена')
        axes[1].set_ylabel('Частота')
        axes[1].set_title('Распределение (тестовая выборка)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Сохранение результатов"""
        print("\n💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 50)
        
        # Создаем папку для результатов
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Сохраняем модель
        model_path = results_dir / "feature_generation_model.pkl"
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"✅ Модель сохранена: {model_path}")
        
        # Сохраняем результаты в JSON
        results_path = results_dir / "feature_generation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Результаты сохранены: {results_path}")
        
        # Сохраняем отчет
        report_path = results_dir / "feature_generation_report.md"
        self._save_report(report_path)
        print(f"✅ Отчет сохранен: {report_path}")
        
        return model_path, results_path, report_path
    
    def _save_report(self, report_path):
        """Сохранение отчета в markdown"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Отчет по генерации признаков и обучению модели\n\n")
            f.write("## Проект: Улучшение baseline-модели для Яндекс Недвижимости\n\n")
            f.write("## Этап 3: Генерация признаков и обучение модели\n\n")
            
            f.write("## Созданные признаки\n\n")
            f.write("### Ручная генерация:\n")
            f.write("- area_per_room - площадь на комнату\n")
            f.write("- total_area_sq - квадрат общей площади\n")
            f.write("- living_to_total_ratio - соотношение жилой и общей площади\n")
            f.write("- kitchen_to_total_ratio - соотношение кухонной и общей площади\n")
            f.write("- building_age - возраст здания\n")
            f.write("- flats_per_floor - плотность застройки\n")
            f.write("- distance_from_center - расстояние от центра\n")
            f.write("- log_price - логарифм цены\n")
            f.write("- is_high_floor, is_low_floor - бинарные признаки этажности\n")
            f.write("- building_type_age - комбинированный признак\n\n")
            
            f.write("### Автоматическая генерация (sklearn):\n")
            f.write("- PolynomialFeatures (степень 2, только взаимодействия)\n")
            f.write("- KBinsDiscretizer (5 бинов, квантильная стратегия)\n")
            f.write("- StandardScaler для числовых признаков\n")
            f.write("- LabelEncoder для категориальных признаков\n\n")
            
            f.write("## Результаты обучения\n\n")
            f.write("### Метрики качества:\n")
            f.write(f"- **Обучающая выборка:**\n")
            f.write(f"  - RMSE: {self.results['train']['rmse']:,.0f}\n")
            f.write(f"  - MAE: {self.results['train']['mae']:,.0f}\n")
            f.write(f"  - R²: {self.results['train']['r2']:.4f}\n")
            f.write(f"  - MAPE: {self.results['train']['mape']:.2f}%\n\n")
            
            f.write(f"- **Тестовая выборка:**\n")
            f.write(f"  - RMSE: {self.results['test']['rmse']:,.0f}\n")
            f.write(f"  - MAE: {self.results['test']['mae']:,.0f}\n")
            f.write(f"  - R²: {self.results['test']['r2']:.4f}\n")
            f.write(f"  - MAPE: {self.results['test']['mape']:.2f}%\n\n")
            
            if 'cross_validation' in self.results:
                f.write("### Кросс-валидация:\n")
                f.write(f"- Средний RMSE: {self.results['cross_validation']['mean_rmse']:,.0f}\n")
                f.write(f"- Стандартное отклонение: {self.results['cross_validation']['std_rmse']:,.0f}\n\n")
            
            f.write("## Архитектура решения\n\n")
            f.write("1. **Ручная генерация признаков** на основе доменной экспертизы\n")
            f.write("2. **Sklearn Pipeline** с ColumnTransformer\n")
            f.write("3. **Автоматическая генерация** полиномиальных признаков\n")
            f.write("4. **Дискретизация** числовых признаков\n")
            f.write("5. **XGBoost** как финальная модель\n")
            f.write("6. **Комплексная оценка** качества модели\n")
    
    def run_complete_feature_generation(self):
        """Запуск полного процесса генерации признаков"""
        print("🚀 ЗАПУСК ПОЛНОГО ПРОЦЕССА ГЕНЕРАЦИИ ПРИЗНАКОВ")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. Загрузка данных
        self.load_data()
        
        # 2. Ручная генерация признаков
        self.manual_feature_engineering()
        
        # 3. Создание sklearn препроцессора
        self.create_sklearn_preprocessor()
        
        # 4. Создание sklearn пайплайна
        self.create_sklearn_pipeline()
        
        # 5. Подготовка данных для обучения
        self.prepare_data_for_training()
        
        # 6. Обучение модели
        self.train_pipeline_model()
        
        # 7. Оценка качества
        self.evaluate_model()
        
        # 8. Кросс-валидация
        self.cross_validation_analysis()
        
        # 9. Анализ важности признаков
        self.feature_importance_analysis()
        
        # 10. Создание визуализаций
        self.create_visualizations()
        
        # 11. Сохранение результатов
        model_path, results_path, report_path = self.save_results()
        
        elapsed_time = time.time() - start_time
        
        print("\n🎉 ГЕНЕРАЦИЯ ПРИЗНАКОВ ЗАВЕРШЕНА УСПЕШНО!")
        print("=" * 80)
        print(f"⏱️ Общее время выполнения: {elapsed_time:.2f} сек")
        print(f"📊 Создано признаков: {self.df_features.shape[1] - self.df.shape[1] + 1}")
        print(f"📁 Модель сохранена: {model_path}")
        print(f"📄 Результаты сохранены: {results_path}")
        print(f"📋 Отчет сохранен: {report_path}")
        
        return self.results


def main():
    """Основная функция"""
    # Путь к данным
    data_path = "data/initial_data_set.csv"
    
    # Создаем генератор признаков
    feature_generator = FeatureGenerator(data_path)
    
    # Запускаем полный процесс
    results = feature_generator.run_complete_feature_generation()
    
    return results


if __name__ == "__main__":
    main()
