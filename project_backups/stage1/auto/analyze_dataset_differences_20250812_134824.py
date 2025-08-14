#!/usr/bin/env python3
"""
Быстрый анализ различий между исходным и улучшенным датасетами
"""

import pandas as pd
import numpy as np

def main():
    print("🔍 АНАЛИЗ РАЗЛИЧИЙ МЕЖДУ ДАТАСЕТАМИ")
    print("=" * 50)
    
    # Загружаем оба датасета
    df1 = pd.read_csv("data/initial_data_set.csv")
    df2 = pd.read_csv("data/merged_data_improved.csv")
    
    print(f"📊 Исходные: {df1.shape[0]:,} × {df1.shape[1]}")
    print(f"📊 Улучшенные: {df2.shape[0]:,} × {df2.shape[1]}")
    
    # Различия в столбцах
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    new_cols = cols2 - cols1
    removed_cols = cols1 - cols2
    
    if new_cols:
        print(f"\n✅ Новые столбцы: {new_cols}")
        for col in new_cols:
            print(f"   {col}: {df2[col].dtype}, уникальных: {df2[col].nunique()}")
            if df2[col].dtype in ['int64', 'float64']:
                print(f"      Диапазон: {df2[col].min():.2f} - {df2[col].max():.2f}")
    
    if removed_cols:
        print(f"\n❌ Удаленные столбцы: {removed_cols}")
    
    # Общие столбцы - изменения в диапазонах
    common_cols = cols1 & cols2
    print(f"\n📋 Общих столбцов: {len(common_cols)}")
    
    print("\n🔍 ИЗМЕНЕНИЯ В ДИАПАЗОНАХ ЗНАЧЕНИЙ:")
    for col in sorted(common_cols):
        if col in df1.columns and col in df2.columns:
            if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']:
                min1, max1 = df1[col].min(), df1[col].max()
                min2, max2 = df2[col].min(), df2[col].max()
                
                if min1 != min2 or max1 != max2:
                    print(f"   {col}:")
                    print(f"      Было: {min1:.1f} - {max1:.1f}")
                    print(f"      Стало: {min2:.1f} - {max2:.1f}")
    
    # Анализ целевой переменной
    target = 'price'
    if target in df1.columns and target in df2.columns:
        print(f"\n🎯 АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ({target}):")
        print(f"   Исходные - среднее: {df1[target].mean():,.0f}, медиана: {df1[target].median():,.0f}")
        print(f"   Улучшенные - среднее: {df2[target].mean():,.0f}, медиана: {df2[target].median():,.0f}")
        
        print(f"   Исходные - min: {df1[target].min():,.0f}, max: {df1[target].max():,.0f}")
        print(f"   Улучшенные - min: {df2[target].min():,.0f}, max: {df2[target].max():,.0f}")

if __name__ == "__main__":
    main()
