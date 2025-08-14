#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Этап 2: Исследовательский анализ данных (EDA)
Проект: Улучшение baseline-модели для Яндекс Недвижимости

Требования проекта:
1. Загрузить данные в Jupyter Notebook, используя библиотеку pandas
2. Провести предварительную обработку данных (пропущенные значения, преобразование типов)
3. Визуализировать данные с помощью matplotlib и seaborn
4. Проанализировать взаимодействие признаков и целевой переменной
5. Создать markdown-ячейку с минимум 3 ключевыми выводами
6. Залогировать в MLflow Jupyter Notebook и markdown-файл с выводами
"""

import os
import warnings
import yaml
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Настройки для корректного отображения русского текста
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
warnings.filterwarnings("ignore")

class EDAAnalyzer:
    """Класс для проведения исследовательского анализа данных"""
    
    def __init__(self, data_path, config_path="../config.yaml"):
        """
        Инициализация анализатора
        
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
        
        # Загружаем конфигурацию
        self._load_config()
        
        # Настройки отображения
        self._setup_display()
        
    def _load_config(self):
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.target_column = self.config['preprocessing']['features']['target_column']
            self.drop_columns = self.config['preprocessing']['features']['drop_columns']
            print("✅ Конфигурация загружена успешно")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            # Значения по умолчанию
            self.target_column = 'price'
            self.drop_columns = ['id', 'building_id']
    
    def _setup_display(self):
        """Настройка параметров отображения"""
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 64
        sns.set_theme(style="ticks", palette="pastel")
        plt.style.use('seaborn-v0_8')
    
    def load_data(self):
        """Загрузка данных с предварительной обработкой"""
        print("�� ЗАГРУЗКА ДАННЫХ...")
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
    
    def basic_data_analysis(self):
        """Базовый анализ структуры данных"""
        print("\n�� БАЗОВЫЙ АНАЛИЗ ДАННЫХ")
        print("=" * 50)
        
        # Информация о данных
        print("�� ИНФОРМАЦИЯ О ДАННЫХ:")
        print(self.df.info())
        
        print("\n�� СТАТИСТИЧЕСКОЕ ОПИСАНИЕ:")
        print(self.df.describe())
        
        print("\n🔢 ТИПЫ ДАННЫХ:")
        print(self.df.dtypes.value_counts())
        
        print("\n❓ ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            missing_percent = (missing_data / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Количество': missing_data,
                'Процент': missing_percent
            }).sort_values('Количество', ascending=False)
            print(missing_df[missing_df['Количество'] > 0])
        else:
            print("✅ Пропущенных значений нет")
        
        print("\n�� ПЕРВЫЕ 5 СТРОК:")
        print(self.df.head())
        
        print("\n🔍 ПОСЛЕДНИЕ 5 СТРОК:")
        print(self.df.tail())
    
    def numerical_features_analysis(self):
        """Анализ числовых признаков"""
        print("\n🔢 АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
        print("=" * 50)
        
        # Получаем числовые признаки из конфигурации
        numerical_features = self.config['preprocessing']['features'].get('numerical_features', [])
        
        if not numerical_features:
            # Если не указаны в конфиге, определяем автоматически
            numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_features = [col for col in numerical_features if col != self.target_column]
        
        print(f"�� Анализируем {len(numerical_features)} числовых признаков:")
        
        for feature in numerical_features:
            if feature in self.df.columns:
                print(f"\n�� {feature.upper()}:")
                print(f"   Тип: {self.df[feature].dtype}")
                print(f"   Уникальных значений: {self.df[feature].nunique()}")
                print(f"   Пропущенных значений: {self.df[feature].isnull().sum()}")
                print(f"   Минимум: {self.df[feature].min()}")
                print(f"   Максимум: {self.df[feature].max()}")
                print(f"   Среднее: {self.df[feature].mean():.2f}")
                print(f"   Медиана: {self.df[feature].median():.2f}")
                print(f"   Стандартное отклонение: {self.df[feature].std():.2f}")
                print(f"   Квартили: {self.df[feature].quantile([0.25, 0.5, 0.75]).tolist()}")
    
    def categorical_features_analysis(self):
        """Анализ категориальных признаков"""
        print("\n🏷️ АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
        print("=" * 50)
        
        # Получаем категориальные признаки из конфигурации
        categorical_features = self.config['preprocessing']['features'].get('categorical_numeric_features', [])
        
        if not categorical_features:
            # Если не указаны в конфиге, определяем автоматически
            categorical_features = self.df.select_dtypes(include=['category', 'object']).columns.tolist()
        
        print(f"�� Анализируем {len(categorical_features)} категориальных признаков:")
        
        for feature in categorical_features:
            if feature in self.df.columns:
                print(f"\n�� {feature.upper()}:")
                print(f"   Тип: {self.df[feature].dtype}")
                print(f"   Уникальных значений: {self.df[feature].nunique()}")
                print(f"   Пропущенных значений: {self.df[feature].isnull().sum()}")
                print(f"   Топ-10 значений:")
                value_counts = self.df[feature].value_counts().head(10)
                for value, count in value_counts.items():
                    percentage = (count / len(self.df)) * 100
                    print(f"     {value}: {count:,} ({percentage:.1f}%)")
    
    def boolean_features_analysis(self):
        """Анализ булевых признаков"""
        print("\n✅ АНАЛИЗ БУЛЕВЫХ ПРИЗНАКОВ")
        print("=" * 50)
        
        # Получаем булевы признаки из конфигурации
        boolean_features = self.config['preprocessing']['features'].get('boolean_features', [])
        
        if not boolean_features:
            # Если не указаны в конфиге, определяем автоматически
            boolean_features = self.df.select_dtypes(include=['bool']).columns.tolist()
        
        print(f"📊 Анализируем {len(boolean_features)} булевых признаков:")
        
        for feature in boolean_features:
            if feature in self.df.columns:
                print(f"\n�� {feature.upper()}:")
                print(f"   Тип: {self.df[feature].dtype}")
                print(f"   Уникальных значений: {self.df[feature].nunique()}")
                print(f"   Пропущенных значений: {self.df[feature].isnull().sum()}")
                print(f"   Распределение:")
                value_counts = self.df[feature].value_counts()
                for value, count in value_counts.items():
                    percentage = (count / len(self.df)) * 100
                    print(f"     {value}: {count:,} ({percentage:.1f}%)")
    
    def target_analysis(self):
        """Анализ целевой переменной"""
        print("\n🎯 АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
        print("=" * 50)
        
        target = self.df[self.target_column]
        
        print(f"📊 {self.target_column.upper()}:")
        print(f"   Тип: {target.dtype}")
        print(f"   Размер: {len(target):,}")
        print(f"   Пропущенных значений: {target.isnull().sum()}")
        print(f"   Минимум: {target.min():,.0f}")
        print(f"   Максимум: {target.max():,.0f}")
        print(f"   Среднее: {target.mean():,.0f}")
        print(f"   Медиана: {target.median():,.0f}")
        print(f"   Стандартное отклонение: {target.std():,.0f}")
        print(f"   Квартили: {target.quantile([0.25, 0.5, 0.75]).tolist()}")
        
        # Анализ выбросов
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = target[(target < lower_bound) | (target > upper_bound)]
        print(f"   Выбросы (IQR метод): {len(outliers):,} ({len(outliers)/len(target)*100:.1f}%)")
        print(f"   Границы выбросов: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    
    def create_visualizations(self):
        """Создание визуализаций"""
        print("\n�� СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("=" * 50)
        
        # Создаем папку для графиков
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Распределение целевой переменной
        self._plot_target_distribution(plots_dir)
        
        # 2. Распределение числовых признаков
        self._plot_numerical_distributions(plots_dir)
        
        # 3. Распределение категориальных признаков
        self._plot_categorical_distributions(plots_dir)
        
        # 4. Корреляционная матрица
        self._plot_correlation_matrix(plots_dir)
        
        # 5. Взаимосвязь признаков с целевой переменной
        self._plot_feature_target_relationships(plots_dir)
        
        print(f"✅ Графики сохранены в папку: {plots_dir}")
    
    def _plot_target_distribution(self, plots_dir):
        """Визуализация распределения целевой переменной"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        target = self.df[self.target_column]
        
        # Основная гистограмма
        sns.histplot(data=self.df, x=self.target_column, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title(f"Распределение {self.target_column}")
        axes[0, 0].set_xlabel(self.target_column)
        axes[0, 0].set_ylabel("Частота")
        
        # Логарифмированная гистограмма
        log_target = np.log1p(target)
        sns.histplot(data=log_target, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title(f"Распределение log({self.target_column} + 1)")
        axes[0, 1].set_xlabel(f"log({self.target_column} + 1)")
        axes[0, 1].set_ylabel("Частота")
        
        # Box plot
        sns.boxplot(data=self.df, x=self.target_column, ax=axes[1, 0])
        axes[1, 0].set_title(f"Box plot {self.target_column}")
        axes[1, 0].set_xlabel(self.target_column)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(target, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot (нормальное распределение)")
        
        plt.tight_layout()
        plt.savefig(plots_dir / "target_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_numerical_distributions(self, plots_dir):
        """Визуализация распределений числовых признаков"""
        numerical_features = self.config['preprocessing']['features'].get('numerical_features', [])
        
        if not numerical_features:
            numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_features = [col for col in numerical_features if col != self.target_column]
        
        # Ограничиваем количество графиков
        features_to_plot = numerical_features[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(features_to_plot):
            if feature in self.df.columns:
                sns.histplot(data=self.df, x=feature, kde=True, ax=axes[i])
                axes[i].set_title(f"Распределение {feature}")
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel("Частота")
        
        # Скрываем лишние оси
        for i in range(len(features_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "numerical_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_categorical_distributions(self, plots_dir):
        """Визуализация распределений категориальных признаков"""
        categorical_features = self.config['preprocessing']['features'].get('categorical_numeric_features', [])
        
        if not categorical_features:
            categorical_features = self.df.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # Ограничиваем количество графиков
        features_to_plot = categorical_features[:4]
        
        if len(features_to_plot) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(features_to_plot):
                if feature in self.df.columns:
                    sns.countplot(data=self.df, x=feature, ax=axes[i])
                    axes[i].set_title(f"Распределение {feature}")
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel("Количество")
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Скрываем лишние оси
            for i in range(len(features_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "categorical_distributions.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_correlation_matrix(self, plots_dir):
        """Визуализация корреляционной матрицы"""
        # Выбираем только числовые признаки
        numerical_df = self.df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) > 1:
            # Вычисляем корреляционную матрицу
            corr_matrix = numerical_df.corr()
            
            # Создаем тепловую карту
            plt.figure(figsize=(12, 10))
            mask = np.zeros_like(corr_matrix)
            mask[np.triu_indices_from(mask)] = True
            
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            
            plt.title("Корреляционная матрица числовых признаков")
            plt.tight_layout()
            plt.savefig(plots_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_feature_target_relationships(self, plots_dir):
        """Визуализация взаимосвязей признаков с целевой переменной"""
        numerical_features = self.config['preprocessing']['features'].get('numerical_features', [])
        
        if not numerical_features:
            numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_features = [col for col in numerical_features if col != self.target_column]
        
        # Ограничиваем количество графиков
        features_to_plot = numerical_features[:4]
        
        if len(features_to_plot) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(features_to_plot):
                if feature in self.df.columns:
                    # Scatter plot
                    sns.scatterplot(data=self.df, x=feature, y=self.target_column, alpha=0.6, ax=axes[i])
                    axes[i].set_title(f"{feature} vs {self.target_column}")
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel(self.target_column)
                    
                    # Добавляем линию тренда
                    z = np.polyfit(self.df[feature], self.df[self.target_column], 1)
                    p = np.poly1d(z)
                    axes[i].plot(self.df[feature], p(self.df[feature]), "r--", alpha=0.8)
            
            # Скрываем лишние оси
            for i in range(len(features_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "feature_target_relationships.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_insights(self):
        """Генерация ключевых выводов на основе EDA"""
        print("\n�� ГЕНЕРАЦИЯ КЛЮЧЕВЫХ ВЫВОДОВ")
        print("=" * 50)
        
        insights = []
        
        # Анализ целевой переменной
        target = self.df[self.target_column]
        target_mean = target.mean()
        target_median = target.median()
        target_std = target.std()
        
        if abs(target_mean - target_median) > target_std * 0.5:
            insights.append("Целевая переменная имеет асимметричное распределение (скошенность), что может потребовать логарифмического преобразования для улучшения качества модели.")
        else:
            insights.append("Целевая переменная имеет близкое к нормальному распределение, что благоприятно для линейных моделей.")
        
        # Анализ корреляций
        numerical_df = self.df.select_dtypes(include=[np.number])
        if len(numerical_df.columns) > 1:
            corr_matrix = numerical_df.corr()
            target_correlations = corr_matrix[self.target_column].abs().sort_values(ascending=False)
            
            top_correlations = target_correlations[1:4]  # Исключаем саму целевую переменную
            if len(top_correlations) > 0:
                top_features = top_correlations.index.tolist()
                insights.append(f"Наиболее сильные корреляции с целевой переменной имеют признаки: {', '.join(top_features)}. Это указывает на их важность для прогнозирования.")
        
        # Анализ категориальных признаков
        categorical_features = self.config['preprocessing']['features'].get('categorical_numeric_features', [])
        if categorical_features:
            insights.append(f"Категориальные признаки ({', '.join(categorical_features)}) требуют специальной обработки (кодирование) перед подачей в модель.")
        
        # Анализ выбросов
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = len(target[(target < Q1 - 1.5 * IQR) | (target > Q3 + 1.5 * IQR)])
        outliers_percent = (outliers_count / len(target)) * 100
        
        if outliers_percent > 5:
            insights.append(f"Обнаружено значительное количество выбросов в целевой переменной ({outliers_percent:.1f}%), что может потребовать специальной обработки или использования робастных алгоритмов.")
        else:
            insights.append("Количество выбросов в данных минимально, что благоприятно для стандартных алгоритмов машинного обучения.")
        
        # Анализ размерности данных
        if self.df.shape[1] > 20:
            insights.append("Высокая размерность признакового пространства может потребовать применения методов отбора признаков для предотвращения переобучения.")
        else:
            insights.append("Умеренная размерность признакового пространства позволяет использовать стандартные алгоритмы без дополнительной регуляризации.")
        
        # Выводим выводы
        print("�� КЛЮЧЕВЫЕ ВЫВОДЫ ПОСЛЕ EDA:")
        print("=" * 50)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
    
    def save_eda_report(self, insights):
        """Сохранение отчета EDA"""
        print("\n💾 СОХРАНЕНИЕ ОТЧЕТА EDA")
        print("=" * 50)
        
        # Создаем папку для отчетов
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Сохраняем markdown отчет
        report_path = reports_dir / "eda_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Отчет по исследовательскому анализу данных (EDA)\n\n")
            f.write("## Проект: Улучшение baseline-модели для Яндекс Недвижимости\n\n")
            f.write("## Датасет\n\n")
            f.write(f"- **Файл:** {self.data_path}\n")
            f.write(f"- **Размер:** {self.df.shape[0]:,} строк × {self.df.shape[1]} столбцов\n")
            f.write(f"- **Целевая переменная:** {self.target_column}\n\n")
            
            f.write("## Ключевые выводы\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n\n")
            
            f.write("## Статистическая сводка\n\n")
            f.write("### Числовые признаки\n\n")
            numerical_features = self.config['preprocessing']['features'].get('numerical_features', [])
            if not numerical_features:
                numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
                numerical_features = [col for col in numerical_features if col != self.target_column]
            
            for feature in numerical_features:
                if feature in self.df.columns:
                    f.write(f"- **{feature}:**\n")
                    f.write(f"  - Среднее: {self.df[feature].mean():.2f}\n")
                    f.write(f"  - Медиана: {self.df[feature].median():.2f}\n")
                    f.write(f"  - Стандартное отклонение: {self.df[feature].std():.2f}\n\n")
            
            f.write("### Категориальные признаки\n\n")
            categorical_features = self.config['preprocessing']['features'].get('categorical_numeric_features', [])
            for feature in categorical_features:
                if feature in self.df.columns:
                    f.write(f"- **{feature}:** {self.df[feature].nunique()} уникальных значений\n\n")
        
        print(f"✅ Отчет сохранен: {report_path}")
        
        return report_path
    
    
    def run_complete_eda(self):
        """Запуск полного EDA анализа"""
        print("�� ЗАПУСК ПОЛНОГО EDA АНАЛИЗА")
        print("=" * 80)
        
        # 1. Загрузка данных
        self.load_data()
        
        # 2. Базовый анализ
        self.basic_data_analysis()
        
        # 3. Анализ признаков
        self.numerical_features_analysis()
        self.categorical_features_analysis()
        self.boolean_features_analysis()
        
        # 4. Анализ целевой переменной
        self.target_analysis()
        
        # 5. Создание визуализаций
        self.create_visualizations()
        
        # 6. Генерация выводов
        insights = self.generate_insights()
        
        # 7. Сохранение отчета
        report_path = self.save_eda_report(insights)
        
        
        print("\n🎉 EDA АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("=" * 80)
        print(f"�� Проанализировано {self.df.shape[0]:,} строк × {self.df.shape[1]} столбцов")
        print(f"📁 Графики сохранены в папку: plots/")
        print(f"📄 Отчет сохранен: {report_path}")
        print(f"�� Сгенерировано {len(insights)} ключевых выводов")
        
        return insights


def main():
    """Основная функция"""
    # Путь к данным
    data_path = "data/initial_data_set.csv"
    
    # Создаем анализатор
    analyzer = EDAAnalyzer(data_path)
    
    # Запускаем полный анализ
    insights = analyzer.run_complete_eda()
    
    return insights


if __name__ == "__main__":
    main()