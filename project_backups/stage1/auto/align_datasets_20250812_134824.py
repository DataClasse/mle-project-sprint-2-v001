#!/usr/bin/env python3
"""
Приведение улучшенного датасета в соответствие с исходным по структуре столбцов
"""

import pandas as pd
import os
from datetime import datetime

def align_datasets():
    """Приводит улучшенный датасет в соответствие с исходным"""
    
    print("🔧 ПРИВЕДЕНИЕ ДАТАСЕТА В СООТВЕТСТВИЕ")
    print("=" * 50)
    
    # Загружаем датасеты
    original_path = "data/initial_data_set.csv"
    improved_path = "data/merged_data_improved.csv"
    
    print(f"📥 Загружаем исходный датасет: {original_path}")
    df_original = pd.read_csv(original_path)
    
    print(f"📥 Загружаем улучшенный датасет: {improved_path}")
    df_improved = pd.read_csv(improved_path)
    
    print(f"\n📊 РАЗМЕРЫ:")
    print(f"   Исходный: {df_original.shape[0]:,} × {df_original.shape[1]}")
    print(f"   Улучшенный: {df_improved.shape[0]:,} × {df_improved.shape[1]}")
    
    # Получаем эталонную структуру из исходного датасета
    target_columns = list(df_original.columns)
    target_dtypes = df_original.dtypes.to_dict()
    
    print(f"\n🎯 ЦЕЛЕВАЯ СТРУКТУРА ({len(target_columns)} столбцов):")
    for i, col in enumerate(target_columns, 1):
        print(f"   {i:2d}. {col} ({target_dtypes[col]})")
    
    # Анализируем текущее состояние улучшенного датасета
    current_columns = list(df_improved.columns)
    
    print(f"\n🔍 ТЕКУЩЕЕ СОСТОЯНИЕ УЛУЧШЕННОГО ({len(current_columns)} столбцов):")
    
    # Находим различия
    missing_cols = set(target_columns) - set(current_columns)
    extra_cols = set(current_columns) - set(target_columns)
    
    if missing_cols:
        print(f"❌ Отсутствующие столбцы: {missing_cols}")
    
    if extra_cols:
        print(f"➕ Лишние столбцы: {extra_cols}")
    
    # Создаем копию для изменений
    df_aligned = df_improved.copy()
    
    print(f"\n🔧 ПРИМЕНЯЕМ ИЗМЕНЕНИЯ:")
    
    # 1. Переименовываем building_type_int в building_type
    if 'building_type_int' in df_aligned.columns and 'building_type' not in df_aligned.columns:
        df_aligned = df_aligned.rename(columns={'building_type_int': 'building_type'})
        print("✅ Переименован building_type_int → building_type")
    
    # 2. Удаляем лишние столбцы
    columns_to_drop = [col for col in df_aligned.columns if col not in target_columns]
    
    if columns_to_drop:
        print(f"🗑️  Удаляем лишние столбцы: {columns_to_drop}")
        df_aligned = df_aligned.drop(columns=columns_to_drop)
    
    # 3. Проверяем наличие всех нужных столбцов
    missing_after_changes = set(target_columns) - set(df_aligned.columns)
    if missing_after_changes:
        print(f"⚠️  Внимание! Отсутствуют столбцы: {missing_after_changes}")
        return None
    
    # 4. Меняем порядок столбцов в соответствии с исходным
    df_aligned = df_aligned[target_columns]
    print("✅ Порядок столбцов приведен в соответствие")
    
    # 5. Проверяем типы данных
    print(f"\n📋 ПРОВЕРКА ТИПОВ ДАННЫХ:")
    type_changes = []
    
    for col in target_columns:
        current_type = df_aligned[col].dtype
        target_type = target_dtypes[col]
        
        if current_type != target_type:
            type_changes.append((col, current_type, target_type))
            
            # Пытаемся привести к нужному типу
            try:
                df_aligned[col] = df_aligned[col].astype(target_type)
                print(f"   ✅ {col}: {current_type} → {target_type}")
            except Exception as e:
                print(f"   ❌ {col}: не удалось преобразовать {current_type} → {target_type} ({e})")
    
    if not type_changes:
        print("   ✅ Все типы данных соответствуют")
    
    # 6. Финальная проверка
    print(f"\n✅ РЕЗУЛЬТАТ:")
    print(f"   Размер: {df_aligned.shape[0]:,} × {df_aligned.shape[1]}")
    print(f"   Столбцы соответствуют исходному: {'✅ Да' if list(df_aligned.columns) == target_columns else '❌ Нет'}")
    
    # Сравниваем структуры
    structure_match = True
    print(f"\n📊 СРАВНЕНИЕ СТРУКТУР:")
    print(f"{'Столбец':<20} {'Исходный':<15} {'Приведенный':<15} {'Статус'}")
    print("-" * 65)
    
    for col in target_columns:
        orig_type = target_dtypes[col]
        aligned_type = df_aligned[col].dtype
        status = "✅" if orig_type == aligned_type else "❌"
        if orig_type != aligned_type:
            structure_match = False
        print(f"{col:<20} {str(orig_type):<15} {str(aligned_type):<15} {status}")
    
    if structure_match:
        print(f"\n🎉 СТРУКТУРЫ ПОЛНОСТЬЮ СООТВЕТСТВУЮТ!")
    else:
        print(f"\n⚠️  Есть различия в типах данных")
    
    return df_aligned

def save_aligned_dataset(df_aligned):
    """Сохраняет приведенный датасет"""
    
    if df_aligned is None:
        print("❌ Нет данных для сохранения")
        return None
    
    # Создаем резервную копию оригинала
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"data/merged_data_improved_backup_{timestamp}.csv"
    
    print(f"\n💾 СОХРАНЕНИЕ:")
    print(f"   📋 Создаем резервную копию: {backup_path}")
    
    # Делаем бэкап
    import shutil
    shutil.copy2("data/merged_data_improved.csv", backup_path)
    
    # Сохраняем приведенный датасет
    output_path = "data/merged_data_improved_aligned.csv"
    df_aligned.to_csv(output_path, index=False)
    print(f"   ✅ Приведенный датасет сохранен: {output_path}")
    
    # Статистика
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"   📊 Размер файла: {file_size:.1f} MB")
    
    return output_path

def validate_alignment():
    """Проверяет корректность приведения"""
    
    print(f"\n🔍 ФИНАЛЬНАЯ ВАЛИДАЦИЯ:")
    
    # Загружаем для сравнения
    df_original = pd.read_csv("data/initial_data_set.csv")
    df_aligned = pd.read_csv("data/merged_data_improved_aligned.csv")
    
    # Проверяем структуру
    structure_ok = list(df_original.columns) == list(df_aligned.columns)
    types_ok = all(df_original[col].dtype == df_aligned[col].dtype for col in df_original.columns)
    
    print(f"   Структура столбцов: {'✅' if structure_ok else '❌'}")
    print(f"   Типы данных: {'✅' if types_ok else '❌'}")
    print(f"   Количество записей: {df_aligned.shape[0]:,}")
    
    if structure_ok and types_ok:
        print(f"\n🎉 ВАЛИДАЦИЯ ПРОЙДЕНА! Датасет готов к использованию")
        return True
    else:
        print(f"\n❌ ВАЛИДАЦИЯ НЕ ПРОЙДЕНА! Требуется дополнительная работа")
        return False

def main():
    """Основная функция"""
    
    # Приводим датасет в соответствие
    df_aligned = align_datasets()
    
    if df_aligned is not None:
        # Сохраняем результат
        output_path = save_aligned_dataset(df_aligned)
        
        if output_path:
            # Валидируем результат
            validation_ok = validate_alignment()
            
            if validation_ok:
                print(f"\n🎯 РЕКОМЕНДАЦИЯ:")
                print(f"   Теперь можно использовать: data/merged_data_improved_aligned.csv")
                print(f"   Вместо: data/initial_data_set.csv")
                print(f"   При этом сохранится совместимость со всеми скриптами")
    
    print(f"\n✅ ПРОЦЕСС ЗАВЕРШЕН")

if __name__ == "__main__":
    main()
