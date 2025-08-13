#!/usr/bin/env python3
"""
Скрипт для обновления ноутбука: замена простого SFS на комбинированный подход
"""

import json
import sys

def update_notebook():
    """Обновляет ноутбук, заменяя SFS код на комбинированный подход"""
    
    # Загружаем ноутбук
    with open('improve_baseline_model.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Ищем ячейку с SFS кодом
    sfs_cell_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and cell['source']:
            source_text = ''.join(cell['source'])
            if 'Sequential Forward Selection с XGBoost' in source_text and 'k_features=5' in source_text:
                sfs_cell_idx = i
                break
    
    if sfs_cell_idx is None:
        print("❌ Не найдена ячейка с SFS кодом")
        return False
    
    print(f"✅ Найдена SFS ячейка: индекс {sfs_cell_idx}")
    
    # Новый код для комбинированного подхода
    new_code = [
        "%%time\n",
        "\n",
        "# 🔬 КОМБИНИРОВАННЫЙ ПОДХОД: Feature Importance + SFS\n",
        "# На основе экспериментов определен как наиболее эффективный метод\n",
        "print(\"🔍 Комбинированный подход Feature Selection с XGBoost\")\n",
        "print(\"📊 Этап 1: Feature Importance для предварительной фильтрации\")\n",
        "\n",
        "# Создаем модель для анализа важности признаков\n",
        "base_model = xgb.XGBRegressor(\n",
        "    n_estimators=300,\n",
        "    max_depth=6,\n",
        "    learning_rate=0.1,\n",
        "    subsample=0.9,\n",
        "    colsample_bytree=0.8,\n",
        "    random_state=42,\n",
        "    n_jobs=1\n",
        ")\n",
        "\n",
        "# Обучаем модель на всех признаках и получаем важность\n",
        "print(\"⏳ Анализ важности признаков...\")\n",
        "base_model.fit(features_train_top10, target_train)\n",
        "\n",
        "# Получаем важность признаков\n",
        "feature_importance = base_model.feature_importances_\n",
        "feature_names = features_train_top10.columns\n",
        "importance_df = pd.DataFrame({\n",
        "    'feature': feature_names,\n",
        "    'importance': feature_importance\n",
        "}).sort_values('importance', ascending=False)\n",
        "\n",
        "# Отбираем топ-20 признаков для дальнейшего SFS\n",
        "top_20_features = importance_df.head(20)['feature'].tolist()\n",
        "print(f\"✅ Отобрано {len(top_20_features)} лучших признаков по важности\")\n",
        "\n",
        "# Этап 2: SFS внутри топ-20 признаков\n",
        "print(\"\\n📊 Этап 2: Sequential Feature Selection внутри топ-20\")\n",
        "print(\"⚙️ Целевое количество признаков: 16 (по результатам экспериментов)\")\n",
        "\n",
        "# Подготавливаем данные только с топ-20 признаками\n",
        "X_train_top20 = features_train_top10[top_20_features]\n",
        "\n",
        "# Создаем SFS для выбора оптимальных 16 признаков\n",
        "sfs_model = xgb.XGBRegressor(\n",
        "    n_estimators=300,\n",
        "    max_depth=6,\n",
        "    learning_rate=0.1,\n",
        "    subsample=0.9,\n",
        "    colsample_bytree=0.8,\n",
        "    random_state=42,\n",
        "    n_jobs=1\n",
        ")\n",
        "\n",
        "sfs = SFS(\n",
        "    sfs_model,\n",
        "    k_features=16,  # Оптимальное число по экспериментам\n",
        "    forward=True,\n",
        "    floating=False,\n",
        "    scoring='neg_root_mean_squared_error',\n",
        "    cv=3,\n",
        "    n_jobs=1\n",
        ")\n",
        "\n",
        "print(\"⏳ Запуск Sequential Feature Selection...\")\n",
        "sfs = sfs.fit(X_train_top20, target_train)\n",
        "\n",
        "# Получаем финальный список выбранных признаков\n",
        "selected_features = list(sfs.k_feature_names_)\n",
        "\n",
        "print(f\"✅ Комбинированный подход завершен!\")\n",
        "print(f\"📊 Выбрано признаков: {len(selected_features)}\")\n",
        "print(f\"🎯 Выбранные признаки:\")\n",
        "for i, feature in enumerate(selected_features, 1):\n",
        "    print(f\"   {i:2d}. {feature}\")\n",
        "\n",
        "# Сохраняем результаты SFS для анализа\n",
        "sfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T\n",
        "display(sfs_df)"
    ]
    
    # Заменяем код в найденной ячейке
    notebook['cells'][sfs_cell_idx]['source'] = new_code
    
    # Теперь ищем и удаляем ячейки с SBS и union logic
    cells_to_remove = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and cell['source']:
            source_text = ''.join(cell['source'])
            # Ищем ячейки связанные с SBS или union features
            if any(keyword in source_text for keyword in [
                'Sequential Backward Selection',
                'sbs = SFS(',
                'top_sbs =',
                'union_features =',
                'interc_features ='
            ]):
                cells_to_remove.append(i)
    
    # Удаляем ячейки в обратном порядке (чтобы индексы не сместились)
    for i in reversed(cells_to_remove):
        print(f"🗑️ Удаляем ячейку {i}: SBS/Union код")
        del notebook['cells'][i]
    
    # Добавляем новую ячейку для использования selected_features
    new_final_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "final_selected_features",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Используем выбранные признаки из комбинированного подхода\n",
            "features_selected = selected_features\n",
            "print(f\"✅ Финальный список признаков: {len(features_selected)}\")\n",
            "print(features_selected)"
        ]
    }
    
    # Находим позицию для вставки (после SFS ячейки)
    insert_position = sfs_cell_idx + 1
    notebook['cells'].insert(insert_position, new_final_cell)
    
    # Сохраняем обновленный ноутбук
    with open('improve_baseline_model.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print("✅ Ноутбук успешно обновлен!")
    print(f"📝 Заменена ячейка {sfs_cell_idx} на комбинированный подход")
    print(f"🗑️ Удалено {len(cells_to_remove)} ячеек с устаревшим кодом")
    print("📊 Добавлена ячейка для работы с выбранными признаками")
    
    return True

if __name__ == "__main__":
    update_notebook()
