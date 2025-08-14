# Описательный алгоритм решения проекта "Улучшение baseline-модели"

## Обзор проекта

**Цель:** Улучшить метрики базовой модели машинного обучения для оценки стоимости недвижимости в Яндекс Недвижимости через систематический подход к feature engineering, отбору признаков и подбору гиперпараметров с использованием MLflow для отслеживания экспериментов.

**Бизнес-задача:** Увеличить точность предсказания стоимости недвижимости для повышения количества успешных сделок на платформе.

---

## Этап 1: Разворачивание MLflow и регистрация baseline-модели

### 1.1 Настройка MLflow инфраструктуры

**Действия:**
1. Экспорт переменных окружения для S3:
   ```bash
   export MLFLOW_S3_ENDPOINT_URL=<ваш_s3_endpoint>
   export AWS_ACCESS_KEY_ID=<ваш_access_key>
   export AWS_SECRET_ACCESS_KEY=<ваш_secret_key>
   export AWS_BUCKET_NAME=<ваш_bucket>
   ```

2. Запуск MLflow сервера:
   ```bash
   mlflow server \
     --backend-store-uri postgresql://<user>:<password>@<host>:<port>/<db> \
     --default-artifact-root s3://<bucket_name>/artifacts \
     --host 127.0.0.1 \
     --port 5000 \
     --no-serve-artifacts
   ```

### 1.2 Регистрация baseline-модели

**Действия:**
1. Загрузка данных из предыдущего этапа
2. Создание эксперимента в MLflow
3. Логирование baseline-модели с параметрами, метриками и сигнатурой
4. Регистрация модели в Model Registry

**Результат:** Baseline-модель зарегистрирована в MLflow Model Registry как версия 1

---

## Этап 2: Исследовательский анализ данных (EDA)

### 2.1 Подготовка среды для EDA

**Действия:**
1. Настройка параметров отображения pandas:
   ```python
   pd.options.display.max_columns = 100
   pd.options.display.max_rows = 64
   ```

2. Настройка стилей графиков:
   ```python
   sns.set_style("white")
   sns.set_theme(style="whitegrid")
   ```

### 2.2 Загрузка и первичный анализ данных

**Действия:**
1. Загрузка данных из PostgreSQL
2. Применение базовых методов pandas:
   - `head()`, `tail()` - просмотр данных
   - `describe()` - статистические характеристики
   - `isnull().sum()` - поиск пропущенных значений
   - `dtypes` - проверка типов данных

### 2.3 Визуализация данных

**Категориальные признаки:**
- Построение barplot для распределения по категориям
- Создание тепловых карт для бинарных признаков
- Анализ таблиц-воронок для комбинаций признаков

**Числовые признаки:**
- Построение histplot с оценкой плотности (kde=True)
- Анализ статистик по времени (среднее, медиана, мода)
- Визуализация трендов и периодичности

**Целевая переменная:**
- Анализ распределения целевой переменной
- Построение графиков зависимости от признаков
- Расчет конверсии по различным сегментам

### 2.4 Формулировка выводов

**Действия:**
1. Создание Markdown-ячейки с минимум 3 ключевыми выводами
2. Логирование Jupyter Notebook в MLflow как артефакт
3. Сохранение выводов в отдельный файл

**Результат:** EDA завершен, артефакты залогированы в MLflow

---

## Этап 3: Генерация признаков

### 3.1 Предобработка данных

**Методы обработки различных типов данных:**

**Числовые признаки:**
- `StandardScaler` - стандартизация
- `MinMaxScaler` - нормализация в диапазон [0,1]
- `RobustScaler` - устойчивость к выбросам
- `QuantileTransformer` - преобразование к равномерному/нормальному распределению

**Категориальные признаки:**
- `OneHotEncoder` - унитарное кодирование
- `LabelEncoder` - порядковое кодирование

**Числовые преобразования:**
- `KBinsDiscretizer` - дискретизация непрерывных признаков
- `SplineTransformer` - создание сплайн-преобразований
- `PolynomialFeatures` - полиномиальные признаки

### 3.2 Создание пайплайна предобработки

**Действия:**
1. Определение типов признаков с помощью `select_dtypes()`
2. Создание `ColumnTransformer` для комбинирования преобразований:
   ```python
   preprocessor = ColumnTransformer(
       transformers=[
           ('num', StandardScaler(), numerical_features),
           ('cat', OneHotEncoder(), categorical_features),
           ('poly', PolynomialFeatures(degree=2), polynomial_features)
       ]
   )
   ```
3. Интеграция в `Pipeline`:
   ```python
   pipeline = Pipeline(steps=[
       ('preprocessor', preprocessor),
       ('model', model)
   ])
   ```

### 3.3 Автоматическая генерация признаков

**Использование библиотеки autofeat:**
```python
af = AutoFeatRegressor(
    feateng_steps=2,
    max_gb=16,
    transformations=["log", "1+", "sqrt", "abs", "exp-"]
)
X_train_fe = af.fit_transform(X_train, y_train)
```

**Возможные преобразования:**
- `"1/"` - обратное значение
- `"exp"` - экспонента
- `"log"` - логарифм
- `"abs"` - абсолютное значение
- `"sqrt"` - квадратный корень
- `"^2", "^3"` - степени
- `"sin", "cos"` - тригонометрические функции

### 3.4 Обучение и логирование модели

**Действия:**
1. Обучение модели на обогащенном наборе признаков
2. Оценка метрик качества на тестовой выборке
3. Логирование в MLflow:
   - Pipeline с предобработкой
   - Параметры модели
   - Метрики качества
   - Код генерации признаков
   - Сигнатура модели
4. Регистрация модели в Model Registry как версия 2

**Результат:** Модель с сгенерированными признаками зарегистрирована

---

## Этап 4: Отбор признаков

### 4.1 Категории методов отбора признаков

**Обученные методы (supervised):**
- **Обёрточные (wrapper):** Sequential Feature Selection
- **Фильтрационные (filter):** корреляция, ANOVA, chi-square
- **Внутренние (intrinsic):** встроенные в алгоритм

**Необученные методы (unsupervised):**
- Поиск дублирующихся и избыточных признаков

**Методы снижения размерности:**
- PCA, t-SNE для сжатия пространства признаков

### 4.2 Применение обёрточных методов

**Sequential Feature Selection с библиотекой mlxtend:**

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Sequential Forward Selection (SFS)
sfs = SFS(
    estimator=RandomForestClassifier(n_estimators=300),
    k_features=10,
    forward=True,
    floating=False,
    scoring='roc_auc',
    cv=4,
    n_jobs=-1
)

# Sequential Backward Selection (SBS)
sbs = SFS(
    estimator=RandomForestClassifier(n_estimators=300),
    k_features=10,
    forward=False,
    floating=False,
    scoring='roc_auc',
    cv=4,
    n_jobs=-1
)
```

### 4.3 Анализ результатов отбора

**Действия:**
1. Отбор признаков методами SFS и SBS
2. Получение пересечения и объединения отобранных признаков:
   ```python
   interc_features = list(set(top_sbs) & set(top_sfs))
   union_features = list(set(top_sbs) | set(top_sfs))
   ```
3. Построение графиков отбора с помощью `plot_sequential_feature_selection`
4. Сохранение результатов в CSV и PNG файлы

### 4.4 Обучение моделей на отобранных признаках

**Действия:**
1. Обучение модели на пересекающихся признаках
2. Обучение модели на объединенных признаках
3. Сравнение метрик качества
4. Логирование всех артефактов в MLflow
5. Регистрация лучшей модели в Model Registry как версия 3

**Результат:** Модель с отобранными признаками зарегистрирована

---

## Этап 5: Подбор гиперпараметров

### 5.1 Методы подбора гиперпараметров

**Grid Search (решетчатый поиск):**
- Систематический перебор всех комбинаций
- Подходит для небольшого количества гиперпараметров
- Гарантирует исследование всей заданной сетки

**Random Search (случайный поиск):**
- Случайный выбор комбинаций из диапазонов
- Эффективнее при большом количестве гиперпараметров
- Быстрее Grid Search при равном бюджете итераций

**Байесовская оптимизация (Optuna):**
- Использует предыдущие результаты для выбора следующих параметров
- Наиболее эффективный метод для сложных пространств поиска

### 5.2 Применение Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
```

### 5.3 Применение Random Search

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [None] + list(np.arange(5, 31, 5)),
    'min_samples_split': np.arange(2, 21),
    'learning_rate': np.logspace(-3, -1, 10)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
```

### 5.4 Применение Optuna

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # Определение пространства поиска
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    
    # Обучение модели
    model = YourModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    model.fit(X_train, y_train)
    
    # Оценка качества
    predictions = model.predict(X_val)
    score = your_metric(y_val, predictions)
    
    return score

# Настройка MLflow Callback
mlflow_callback = MLflowCallback(
    tracking_uri="http://localhost:5000",
    metric_name="rmse"
)

# Оптимизация
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, callbacks=[mlflow_callback])
```

### 5.5 Методы suggest в Optuna

**Для различных типов гиперпараметров:**
- `suggest_categorical()` - категориальные значения ('adam', 'sgd')
- `suggest_int()` - целые числа в диапазоне
- `suggest_float()` - вещественные числа
- `suggest_loguniform()` - логарифмическое распределение
- `suggest_discrete_uniform()` - дискретные значения с шагом

### 5.6 Финальное обучение и регистрация

**Действия:**
1. Применение обоих методов оптимизации (обязательно Optuna + один из Grid/Random Search)
2. Сравнение результатов оптимизации
3. Обучение финальной модели с лучшими гиперпараметрами
4. Оценка на тестовой выборке
5. Логирование в MLflow:
   - Процесс подбора гиперпараметров
   - Лучшие параметры
   - Финальные метрики
   - Сравнение с предыдущими версиями
6. Регистрация финальной модели в Model Registry как версия 4

**Результат:** Финальная оптимизированная модель зарегистрирована

---

## Критерии успешного выполнения проекта

### Обязательные требования:

1. **Минимум 4 версии модели в MLflow Model Registry:**
   - Версия 1: Baseline модель
   - Версия 2: Модель с генерацией признаков
   - Версия 3: Модель с отбором признаков
   - Версия 4: Модель с подбором гиперпараметров

2. **Логирование в MLflow для каждого этапа:**
   - Артефакты (код, графики, данные)
   - Параметры модели
   - Метрики качества
   - Окружение (environment)

3. **Структурированный код:**
   - Четкая документация
   - Комментарии к ключевым блокам
   - Использование sklearn.Pipeline

4. **Jupyter Notebook:**
   - Все этапы проекта в одном файле
   - Четкая структура и пояснения

5. **Репозиторий:**
   - README.md с описанием проекта
   - requirements.txt или conda.yaml
   - Shell-скрипт для запуска MLflow сервисов

---

## Рекомендации по выполнению

### Организация работы:
1. Выполняйте этапы последовательно
2. Регулярно сохраняйте промежуточные результаты
3. Документируйте каждый шаг
4. Проводите валидацию результатов на каждом этапе

### Техническая часть:
1. Используйте фиксированные random_state для воспроизводимости
2. Регулярно сохраняйте состояние в MLflow
3. Мониторьте использование ресурсов
4. Создавайте резервные копии важных результатов

### Анализ результатов:
1. Сравнивайте метрики между версиями моделей
2. Анализируйте важность признаков
3. Оценивайте вычислительную сложность
4. Формулируйте выводы о влиянии каждого этапа на качество модели

Этот алгоритм обеспечивает систематический подход к улучшению baseline-модели с полным контролем экспериментов через MLflow и применением современных методов feature engineering и оптимизации гиперпараметров.
