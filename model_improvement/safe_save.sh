#!/bin/bash
# УНИВЕРСАЛЬНЫЙ СКРИПТ БЕЗОПАСНОГО СОХРАНЕНИЯ
# Адаптирован для многоэтапной структуры проекта
# Автор: AI Assistant для защиты кода Дмитрия
# ================================================

set -e

# Подключаем детектор этапов
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "$PROJECT_ROOT/project_safety_detector.sh"

echo "🛡️  УНИВЕРСАЛЬНАЯ СИСТЕМА БЕЗОПАСНОСТИ"
echo "======================================"

# Получаем информацию о текущем этапе
if ! project_safety_check; then
    echo "❌ Ошибка определения этапа проекта"
    exit 1
fi

echo

backup_and_commit() {
    local stage="$PROJECT_STAGE"
    local backup_path="$(get_backup_path $stage auto)"
    
    # Создаем директорию для backup'ов
    mkdir -p "$backup_path"
    
    # Получаем список файлов для защиты
    local files_to_protect=($(get_files_to_protect "$stage"))
    
    if [ ${#files_to_protect[@]} -eq 0 ]; then
        echo "⚠️  Нет файлов для защиты в текущей директории"
        return 1
    fi
    
    local backup_count=0
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo "📋 Сохраняю файлы этапа: $(get_stage_description $stage)"
    
    # Обрабатываем каждый файл
    for file in "${files_to_protect[@]}"; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            local extension="${filename##*.}"
            local basename="${filename%.*}"
            local backup_name="${basename}_${timestamp}.${extension}"
            
            cp "$file" "$backup_path/$backup_name"
            echo "✅ Backup: $file → $backup_name"
            
            # Добавляем в Git
            git add "$file"
            backup_count=$((backup_count + 1))
        fi
    done
    
    # Коммитим изменения
    if git commit -m "[$stage] Auto backup - $(date '+%Y-%m-%d %H:%M:%S')"; then
        echo "✅ Изменения сохранены в Git (файлов: $backup_count)"
    else
        echo "📝 Нет новых изменений для коммита"
    fi
    
    # Статистика
    show_backup_stats "$backup_path" "$backup_count"
    
    echo "✅ Этап '$stage' сохранен безопасно!"
}

# Статистика backup'ов
show_backup_stats() {
    local backup_path="$1"
    local current_count="$2"
    
    if [ -d "$backup_path" ]; then
        local total_backups=$(find "$backup_path" -type f | wc -l)
        local total_size=$(du -sh "$backup_path" 2>/dev/null | cut -f1)
        
        echo "📊 СТАТИСТИКА BACKUP'ОВ:"
        echo "   📁 Путь: $backup_path"
        echo "   📝 Всего backup'ов: $total_backups"
        echo "   💾 Размер: $total_size"
        echo "   ⏰ Последний: $(date '+%H:%M:%S')"
    fi
}

# Очистка старых backup'ов (универсальная)
cleanup_backups() {
    local stage="$PROJECT_STAGE"
    local backup_path="$(get_backup_path $stage auto)"
    
    if [ -d "$backup_path" ]; then
        echo "🧹 Очистка старых backup'ов (>7 дней)..."
        
        # Удаляем файлы старше 7 дней
        local deleted_count=$(find "$backup_path" -type f -mtime +7 -delete -print 2>/dev/null | wc -l)
        
        if [ "$deleted_count" -gt 0 ]; then
            echo "✅ Удалено старых backup'ов: $deleted_count"
        else
            echo "📝 Нет старых backup'ов для удаления"
        fi
        
        # Показываем оставшееся количество
        local remaining=$(find "$backup_path" -type f | wc -l)
        echo "📊 Осталось backup'ов: $remaining"
    fi
}

# Главная функция
main() {
    echo "🕐 Запуск: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    
    backup_and_commit
    echo
    cleanup_backups
    
    echo
    echo "======================================"
    echo "🚀 Готово! Можете безопасно работать"
    echo "💡 Используйте '../project_safety_detector.sh' для проверки статуса"
}

# Если скрипт запущен напрямую
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
