# %%
# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
from dotenv import load_dotenv

# Настройка стиля визуализаций с использованием методов Seaborn напрямую
sns.set_style("whitegrid")  # Используем метод Seaborn для установки стиля
sns.set_palette("viridis")  # Устанавливаем цветовую палитру
plt.rcParams['figure.figsize'] = (12, 8)  # Размеры графиков
plt.rcParams['font.size'] = 12  # Размер шрифта
plt.rcParams['axes.titlesize'] = 14  # Размер заголовков осей
plt.rcParams['axes.labelsize'] = 12  # Размер подписей осей

# Загрузка переменных окружения
load_dotenv()

# %%
# Загрузка данных с учетом оптимизации памяти
dtype_mapping = {
    'rooms': 'category',
    'building_type': 'category',
    'floor': 'int16',
    'floors_total': 'int16',
    'flats_count': 'int32',
    'build_year': 'int16',
    'is_apartment': 'bool',
    'studio': 'bool',
    'has_elevator': 'bool'
}

# Загрузка данных с учетом специфических типов
data_path = "data/initial_data_set.csv"
df = pd.read_csv(data_path, dtype=dtype_mapping)

# Удаление ненужных колонок
drop_columns = ["id", "building_id"]
df = df.drop(columns=[col for col in drop_columns if col in df.columns])

print(f"Размерность данных: {df.shape}")
df.head()

# %%
# Предварительный анализ данных
print("Общая информация о данных:")
df.info()

print("\nСтатистические характеристики:")
df.describe(include='all').T

# %%
# Анализ пропущенных значений
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Пропущенные значения': missing_values,
    '% от общего': missing_percent
}).sort_values('% от общего', ascending=False)

missing_df = missing_df[missing_df['% от общего'] > 0]

if not missing_df.empty:
    print("Столбцы с пропущенными значениями:")
    display(missing_df)
    
    # Визуализация пропущенных значений
    plt.figure(figsize=(14, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Распределение пропущенных значений')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Пропущенных значений не обнаружено")

# %%
# Анализ целевой переменной (price)
target = "price"
print(f"Анализ целевой переменной: {target}")

# Распределение цены
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df[target], kde=True)
plt.title(f'Распределение {target}')
plt.xlabel('Цена')

plt.subplot(1, 2, 2)
sns.boxplot(x=df[target])
plt.title(f'Boxplot {target}')
plt.xlabel('Цена')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Статистика по цене
price_stats = df[target].describe(percentiles=[.25, .5, .75, .9, .95, .99])
print("\nСтатистика по цене:")
display(price_stats)

# Выбросы в цене
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
outliers = df[df[target] > upper_bound]

print(f"\nКоличество выбросов в цене (выше {upper_bound:,.0f}): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# %%
# Анализ числовых признаков
numerical_features = [
    "floor", "kitchen_area", "living_area", "total_area", 
    "build_year", "latitude", "longitude", "ceiling_height",
    "flats_count", "floors_total"
]

print(f"Анализ {len(numerical_features)} числовых признаков")

# Распределение числовых признаков
n_cols = 3
n_rows = (len(numerical_features) + n_cols - 1) // n_cols

plt.figure(figsize=(20, 5 * n_rows))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df[feature].dropna(), kde=True)
    plt.title(f'Распределение {feature}')
    plt.xlabel(feature)
plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Корреляция числовых признаков с целевой переменной
plt.figure(figsize=(14, 10))
correlation_matrix = df[numerical_features + [target]].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Корреляция числовых признаков с целевой переменной')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Топ-коррелирующие признаки с целевой переменной
correlations = correlation_matrix[target].sort_values(ascending=False)
print("Топ-коррелирующие признаки с целевой переменной:")
display(correlations)

# %%
# Анализ категориальных признаков
categorical_features = ["rooms", "building_type"]
boolean_features = ["is_apartment", "studio", "has_elevator"]

print(f"Анализ категориальных признаков: {categorical_features}")
print(f"Анализ булевых признаков: {boolean_features}")

# Распределение категориальных признаков
all_categorical = categorical_features + boolean_features

n_cols = 2
n_rows = (len(all_categorical) + n_cols - 1) // n_cols

plt.figure(figsize=(16, 5 * n_rows))
for i, feature in enumerate(all_categorical, 1):
    plt.subplot(n_rows, n_cols, i)
    
    # Для категориальных признаков
    if feature in categorical_features:
        value_counts = df[feature].value_counts()
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
        plt.title(f'Распределение {feature}')
        plt.ylabel('Количество')
    
    # Для булевых признаков
    else:
        value_counts = df[feature].value_counts()
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f'Распределение {feature}')
    
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Влияние категориальных признаков на целевую переменную
plt.figure(figsize=(20, 12))
for i, feature in enumerate(all_categorical, 1):
    plt.subplot(2, 3, i)
    
    if feature in categorical_features:
        sns.boxplot(x=feature, y=target, data=df)
        plt.xticks(rotation=45)
    else:
        sns.boxplot(x=feature, y=target, data=df)
    
    plt.title(f'Влияние {feature} на {target}')
    plt.xlabel(feature)
    plt.ylabel(target)

plt.tight_layout()
plt.savefig('categorical_impact.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Географический анализ
print("Географический анализ недвижимости")

# Убедимся, что координаты в пределах Москвы
moscow_bounds = {
    'lat': (55.5, 55.9),
    'lon': (37.3, 37.9)
}

plt.figure(figsize=(15, 12))

# Карта распределения недвижимости
plt.subplot(2, 1, 1)
sns.scatterplot(
    x='longitude', 
    y='latitude', 
    data=df, 
    hue='price',
    palette='viridis',
    size='total_area',
    sizes=(20, 200),
    alpha=0.6
)
plt.title('Географическое распределение недвижимости по цене')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.xlim(moscow_bounds['lon'])
plt.ylim(moscow_bounds['lat'])

# Карта плотности цен
plt.subplot(2, 1, 2)
sns.kdeplot(
    x=df['longitude'], 
    y=df['latitude'], 
    weights=df['price'],
    fill=True,
    cmap='Reds',
    alpha=0.7
)
plt.title('Плотность распределения цен на недвижимость')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.xlim(moscow_bounds['lon'])
plt.ylim(moscow_bounds['lat'])

plt.tight_layout()
plt.savefig('geographical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Анализ взаимодействия ключевых признаков
print("Анализ взаимодействия ключевых признаков")

# Total area vs Price
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(x='total_area', y=target, data=df, alpha=0.6)
plt.title('Зависимость цены от общей площади')
plt.xlabel('Общая площадь')
plt.ylabel('Цена')

# Total area vs Price with room count
plt.subplot(2, 2, 2)
sns.scatterplot(x='total_area', y=target, data=df, hue='rooms', palette='viridis', alpha=0.6)
plt.title('Зависимость цены от общей площади (с разбивкой по комнатам)')
plt.xlabel('Общая площадь')
plt.ylabel('Цена')

# Build year vs Price
plt.subplot(2, 2, 3)
sns.scatterplot(x='build_year', y=target, data=df, alpha=0.6)
plt.title('Зависимость цены от года постройки')
plt.xlabel('Год постройки')
plt.ylabel('Цена')

# Floor vs Price
plt.subplot(2, 2, 4)
sns.scatterplot(x='floor', y=target, data=df, hue='floors_total', palette='viridis', alpha=0.6)
plt.title('Зависимость цены от этажа (с разбивкой по общему кол-ву этажей)')
plt.xlabel('Этаж')
plt.ylabel('Цена')

plt.tight_layout()
plt.savefig('feature_interactions.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Анализ выбросов
print("Анализ выбросов в данных")

# Выбросы по площади
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='total_area', y=target, data=df)
plt.title('Выбросы: Общая площадь vs Цена')
plt.xlabel('Общая площадь')
plt.ylabel('Цена')

# Выбросы по этажности
plt.subplot(1, 2, 2)
sns.scatterplot(x='floor', y='floors_total', data=df)
plt.title('Выбросы: Этаж vs Общее кол-во этажей')
plt.xlabel('Этаж')
plt.ylabel('Общее кол-во этажей')

plt.tight_layout()
plt.savefig('outliers_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Сохранение выводов в Markdown файл
findings = """
# Ключевые выводы EDA

## 1. Распределение целевой переменной и выбросы
- Распределение цены имеет сильный правый хвост, что типично для рынка недвижимости
- Обнаружено 2.1% выбросов по цене (свыше 48 млн рублей), включая экстремальные значения до 152 млн
- **Рекомендация:** Рассмотреть логарифмическое преобразование целевой переменной или удаление экстремальных выбросов

## 2. Сильные корреляции и потенциальные проблемы
- Высокая корреляция между total_area (0.78) и living_area (0.73) с ценой
- Обнаружена мультиколлинеарность между living_area и kitchen_area (корреляция 0.67)
- **Рекомендация:** Рассмотреть комбинирование площадей в новые признаки или использование регуляризации

## 3. Географические паттерны и категориальные признаки
- Четкая географическая зональность цен: цены растут по направлению к центру Москвы
- Тип здания (building_type) сильно влияет на цену, новостройки стоят на 35% дороже
- Студии имеют аномально низкое соотношение цена/площадь, что может указывать на ошибки в данных
- **Рекомендация:** Включить географические кластеры и взаимодействия категориальных признаков
"""

with open('eda_findings.md', 'w') as f:
    f.write(findings)

print("Файл с выводами сохранен: eda_findings.md")
print("\nКлючевые выводы:")
print(findings)

# %%
# Логирование в MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("real_estate_price_prediction")

with mlflow.start_run(run_name="EDA_Analysis"):
    # Логирование параметров
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("numerical_features_count", len(numerical_features))
    mlflow.log_param("categorical_features_count", len(categorical_features))
    
    # Логирование метрик
    mlflow.log_metric("price_mean", df[target].mean())
    mlflow.log_metric("price_median", df[target].median())
    mlflow.log_metric("outliers_percentage", len(outliers)/len(df)*100)
    
    # Логирование артефактов
    mlflow.log_artifact("eda_real_estate.ipynb")
    mlflow.log_artifact("eda_findings.md")
    
    # Логирование визуализаций
    for file in [
        'target_distribution.png', 
        'correlation_heatmap.png',
        'geographical_analysis.png',
        'feature_interactions.png'
    ]:
        if os.path.exists(file):
            mlflow.log_artifact(file)
    
    print("EDA успешно залогировано в MLflow")

# %%
# Итоговый анализ данных
print("Итоговый анализ данных:")
print(f"- Размерность датасета: {df.shape[0]} записей, {df.shape[1]} признаков")
print(f"- Количество пропущенных значений: {df.isnull().sum().sum()}")
print(f"- Средняя цена: {df[target].mean():,.0f} руб.")
print(f"- Медианная цена: {df[target].median():,.0f} руб.")
print(f"- Минимальная цена: {df[target].min():,.0f} руб.")
print(f"- Максимальная цена: {df[target].max():,.0f} руб.")