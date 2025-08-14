#!/bin/bash
# УНИВЕРСАЛЬНЫЙ ДЕТЕКТОР ЭТАПОВ ПРОЕКТА
# Определяет текущий этап и настройки безопасности
# =====================================================

# Константы проекта
PROJECT_ROOT="/home/mle-user/mle_projects/mle-project-sprint-2-v001"
BACKUP_ROOT="$PROJECT_ROOT/project_backups"

# Функции для определения этапа
detect_project_stage() {
    local current_dir="$(pwd)"
    
    # Проверяем, находимся ли мы в проекте
    if [[ "$current_dir" != "$PROJECT_ROOT"* ]]; then
        echo "ERROR: Не в директории проекта!"
        return 1
    fi
    
    # Определяем этап по директории
    if [[ "$current_dir" == *"/mlflow_server"* ]]; then
        echo "stage1"
    elif [[ "$current_dir" == *"/model_improvement"* ]]; then
        echo "stage2-5"
    elif [[ "$current_dir" == "$PROJECT_ROOT" ]]; then
        echo "root"
    else
        echo "unknown"
    fi
}

# Получение файлов для защиты в зависимости от этапа
get_files_to_protect() {
    local stage="$1"
    local current_dir="$(pwd)"
    
    case "$stage" in
        "stage1")
            # 1-й этап: Python файлы MLflow
            find . -maxdepth 1 -name "*.py" -type f
            ;;
        "stage2-5") 
            # 2-5 этапы: Jupyter notebooks и Python файлы
            find . -maxdepth 1 \( -name "*.ipynb" -o -name "*.py" \) -type f
            ;;
        "root")
            # Корень проекта: все важные файлы
            find . -maxdepth 2 \( -name "*.ipynb" -o -name "*.py" -o -name "*.md" \) -type f | grep -v venv | grep -v backups
            ;;
        *)
            echo "Неизвестный этап: $stage" >&2
            return 1
            ;;
    esac
}

# Создание структуры бэкапов
setup_backup_structure() {
    local stage="$1"
    
    # Создаем основную структуру
    mkdir -p "$BACKUP_ROOT"/{stage1,stage2-5,root,daily_snapshots}
    
    # Создаем подструктуру для текущего этапа
    case "$stage" in
        "stage1")
            mkdir -p "$BACKUP_ROOT/stage1"/{manual,auto,pre_commit}
            ;;
        "stage2-5")
            mkdir -p "$BACKUP_ROOT/stage2-5"/{manual,auto,pre_commit,experiments}
            ;;
        "root")
            mkdir -p "$BACKUP_ROOT/root"/{manual,auto,pre_commit}
            ;;
    esac
}

# Получение пути для бэкапа
get_backup_path() {
    local stage="$1"
    local backup_type="$2"  # manual, auto, pre_commit, experiment
    
    echo "$BACKUP_ROOT/$stage/$backup_type"
}

# Получение описания этапа
get_stage_description() {
    local stage="$1"
    
    case "$stage" in
        "stage1")
            echo "🚀 Этап 1: Настройка MLflow Server"
            ;;
        "stage2-5")
            echo "🧪 Этапы 2-5: Улучшение модели"
            ;;
        "root")
            echo "📁 Корень проекта: Общие настройки"
            ;;
        *)
            echo "❓ Неизвестный этап"
            ;;
    esac
}

# Главная функция проверки
project_safety_check() {
    local stage=$(detect_project_stage)
    
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    echo "🛡️  ПРОЕКТ: mle-project-sprint-2-v001"
    echo "📍 ЭТАП: $(get_stage_description $stage)"
    echo "📂 ДИРЕКТОРИЯ: $(pwd)"
    echo "💾 БЭКАПЫ: $BACKUP_ROOT/$stage/"
    
    # Настраиваем структуру бэкапов
    setup_backup_structure "$stage"
    
    # Показываем файлы для защиты
    echo
    echo "📋 ФАЙЛЫ ДЛЯ ЗАЩИТЫ:"
    get_files_to_protect "$stage" | sed 's/^/   /'
    
    # Экспортируем переменные для других скриптов
    export PROJECT_STAGE="$stage"
    export PROJECT_ROOT="$PROJECT_ROOT"
    export BACKUP_ROOT="$BACKUP_ROOT"
    export CURRENT_BACKUP_PATH="$(get_backup_path $stage auto)"
    
    return 0
}

# Если скрипт запущен напрямую
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    project_safety_check "$@"
fi
