#!/bin/bash
# УНИВЕРСАЛЬНЫЙ ЕЖЕДНЕВНЫЙ WORKFLOW ДЛЯ МНОГОЭТАПНОГО ML ПРОЕКТА
# Адаптирован для работы с этапами: mlflow_server + model_improvement
# Система защиты кода для Дмитрия
# =================================================================

set -e  # Остановка при ошибках

# Подключаем детектор этапов
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "$PROJECT_ROOT/project_safety_detector.sh"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Функция для красивого вывода
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${PURPLE}🛡️  $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. УТРЕННЯЯ НАСТРОЙКА РАБОЧЕГО ДНЯ
morning_setup() {
    print_header "УТРЕННЯЯ НАСТРОЙКА МНОГОЭТАПНОГО ПРОЕКТА"
    
    # Определяем текущий этап
    if ! project_safety_check; then
        print_error "Ошибка определения этапа проекта"
        return 1
    fi
    
    local stage="$PROJECT_STAGE"
    echo
    
    # Переходим в корень проекта для Git операций
    cd "$PROJECT_ROOT"
    
    # Проверяем Git статус
    if git status --porcelain | grep -q .; then
        print_warning "Есть незакоммиченные изменения:"
        git status --short
        echo
        read -p "Хотите сохранить их? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git add -A
            git commit -m "[$stage] Morning cleanup - $(date '+%Y-%m-%d %H:%M')"
            print_success "Изменения сохранены"
        fi
    fi
    
    # Создаем новую рабочую ветку для дня
    local branch_name="work/$(date +%Y-%m-%d)"
    if ! git show-ref --verify --quiet refs/heads/"$branch_name"; then
        git checkout -b "$branch_name"
        print_success "Создана рабочая ветка: $branch_name"
    else
        git checkout "$branch_name"
        print_success "Переключились на рабочую ветку: $branch_name"
    fi
    
    # Синхронизация с remote
    if git remote | grep -q origin; then
        git fetch origin
        print_success "Синхронизация с удаленным репозиторием"
    fi
    
    # Создаем структуру backup'ов для всех этапов
    setup_backup_structure "$stage"
    print_success "Структура backup'ов подготовлена"
    
    # Возвращаемся в рабочую директорию
    cd - > /dev/null
    
    print_success "Этап '$stage' готов к работе! $(date '+%H:%M')"
    print_warning "Используйте './safe_save.sh' для регулярного сохранения"
}

# 2. АВТОМАТИЧЕСКИЙ BACKUP КАЖДЫЙ ЧАС
hourly_backup() {
    print_header "ЕЖЕЧАСНЫЙ BACKUP"
    
    # Определяем текущий этап
    if ! project_safety_check; then
        print_error "Ошибка определения этапа проекта"
        return 1
    fi
    
    local stage="$PROJECT_STAGE"
    local snapshot_path="$BACKUP_ROOT/daily_snapshots"
    local timestamp=$(date +%Y%m%d_%H%M)
    
    echo
    mkdir -p "$snapshot_path"
    
    # Получаем файлы для backup'а
    local files_to_backup=($(get_files_to_protect "$stage"))
    local backup_count=0
    
    if [ ${#files_to_backup[@]} -eq 0 ]; then
        print_warning "Нет файлов для backup'а в текущем этапе"
        return 1
    fi
    
    print_success "Создаю snapshot для этапа: $(get_stage_description $stage)"
    
    # Создаем snapshot для каждого файла
    for file in "${files_to_backup[@]}"; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            local extension="${filename##*.}"
            local basename="${filename%.*}"
            local snapshot_name="${stage}_${basename}_snapshot_${timestamp}.${extension}"
            
            cp "$file" "$snapshot_path/$snapshot_name"
            print_success "Snapshot: $file → $snapshot_name"
            backup_count=$((backup_count + 1))
        fi
    done
    
    # Переходим в корень для Git операций
    cd "$PROJECT_ROOT"
    
    # Git commit с временной меткой
    git add -A
    if git commit -m "[$stage] Hourly backup - $timestamp"; then
        print_success "Код сохранен в Git: $timestamp"
    else
        print_success "Нет изменений с последнего backup'а"
    fi
    
    # Возвращаемся в рабочую директорию
    cd - > /dev/null
    
    print_success "Файлов в backup'е: $backup_count | Время: $(date '+%H:%M')"
}

# 3. БЕЗОПАСНОЕ СОХРАНЕНИЕ ПЕРЕД ЭКСПЕРИМЕНТАМИ
safe_experiment() {
    print_header "ПОДГОТОВКА К ЭКСПЕРИМЕНТУ"
    
    # Определяем текущий этап
    if ! project_safety_check; then
        print_error "Ошибка определения этапа проекта"
        return 1
    fi
    
    local stage="$PROJECT_STAGE"
    local experiment_name="$1"
    
    echo
    
    if [ -z "$experiment_name" ]; then
        read -p "Название эксперимента: " experiment_name
    fi
    
    local experiment_path="$(get_backup_path $stage experiments)"
    mkdir -p "$experiment_path"
    
    # Переходим в корень для Git операций
    cd "$PROJECT_ROOT"
    
    # Создаем точку восстановления
    git add -A
    git commit -m "[$stage] Before experiment: $experiment_name - $(date)"
    
    # Создаем backup всех файлов этапа
    local files_to_backup=($(get_files_to_protect "$stage"))
    local timestamp=$(date +%Y%m%d_%H%M)
    
    # Возвращаемся в рабочую директорию для backup'а файлов
    cd - > /dev/null
    
    for file in "${files_to_backup[@]}"; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            local extension="${filename##*.}"
            local basename="${filename%.*}"
            local backup_name="${basename}_before_${experiment_name// /_}_${timestamp}.${extension}"
            
            cp "$file" "$experiment_path/$backup_name"
            print_success "Backup: $file → $backup_name"
        fi
    done
    
    # Переходим в корень для создания ветки
    cd "$PROJECT_ROOT"
    
    # Создаем ветку для эксперимента
    local branch_name="experiment/${stage}/${experiment_name// /_}"
    git checkout -b "$branch_name" 2>/dev/null || git checkout "$branch_name"
    
    # Возвращаемся в рабочую директорию
    cd - > /dev/null
    
    print_success "Готов к эксперименту: $experiment_name"
    print_success "Этап: $(get_stage_description $stage)"
    print_success "Backup'ы: $experiment_path"
    print_success "Ветка: $branch_name"
    
    echo
    print_warning "ДЛЯ ОТКАТА ИСПОЛЬЗУЙТЕ:"
    echo "cd $PROJECT_ROOT && git checkout work/$(date +%Y-%m-%d)"
    echo "git reset --hard HEAD~1"
}

# 4. ВЕЧЕРНЕЕ СОХРАНЕНИЕ
evening_save() {
    print_header "ВЕЧЕРНЕЕ СОХРАНЕНИЕ МНОГОЭТАПНОГО ПРОЕКТА"
    
    # Определяем текущий этап
    if ! project_safety_check; then
        print_error "Ошибка определения этапа проекта"
        return 1
    fi
    
    local stage="$PROJECT_STAGE"
    local daily_backup_path="$(get_backup_path $stage manual)"
    
    echo
    mkdir -p "$daily_backup_path"
    
    # Финальный backup всех файлов этапа
    local files_to_backup=($(get_files_to_protect "$stage"))
    local date_stamp=$(date +%Y%m%d)
    local timestamp=$(date +%Y%m%d_%H%M)
    
    print_success "Создаю финальный backup для: $(get_stage_description $stage)"
    
    for file in "${files_to_backup[@]}"; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            local extension="${filename##*.}"
            local basename="${filename%.*}"
            local backup_name="${basename}_daily_${timestamp}.${extension}"
            
            cp "$file" "$daily_backup_path/$backup_name"
            print_success "Финальный backup: $file → $backup_name"
        fi
    done
    
    # Переходим в корень для Git операций
    cd "$PROJECT_ROOT"
    
    # Коммитим все изменения
    git add -A
    git commit -m "[$stage] End of day save - $(date '+%Y-%m-%d %H:%M')"
    
    # Переключаемся на main и мержим
    git checkout main
    git merge "work/$(date +%Y-%m-%d)" --no-ff -m "[$stage] Daily work merge - $(date '+%Y-%m-%d')"
    
    # Пушим в remote (если есть)
    if git remote | grep -q origin; then
        git push origin main
        print_success "Код загружен в удаленный репозиторий"
    fi
    
    # Статистика дня
    local commits_today=$(git log --oneline --since="1 day ago" | wc -l)
    local backups_today=$(find "$BACKUP_ROOT" -name "*${date_stamp}*" -type f | wc -l)
    local project_size=$(du -sh "$PROJECT_ROOT" | cut -f1)
    
    print_success "Коммитов за день: $commits_today"
    print_success "Backup'ов создано: $backups_today"
    print_success "Размер проекта: $project_size"
    print_success "Этап завершен: $(get_stage_description $stage)"
    
    # Возвращаемся в рабочую директорию
    cd - > /dev/null
    
    echo
    print_header "ДЕНЬ ЗАВЕРШЕН УСПЕШНО! 🎉"
    print_warning "Все этапы проекта защищены backup'ами"
}

# 5. ЭКСТРЕННОЕ ВОССТАНОВЛЕНИЕ
emergency_restore() {
    print_header "ЭКСТРЕННОЕ ВОССТАНОВЛЕНИЕ"
    
    # Определяем текущий этап
    if ! project_safety_check; then
        print_error "Ошибка определения этапа проекта"
        return 1
    fi
    
    local stage="$PROJECT_STAGE"
    echo
    
    print_success "Доступные backup'ы для этапа: $(get_stage_description $stage)"
    echo
    
    # Показываем backup'ы из всех категорий
    for backup_type in auto manual pre_commit experiments; do
        local backup_path="$(get_backup_path $stage $backup_type)"
        if [ -d "$backup_path" ] && [ "$(ls -A $backup_path 2>/dev/null)" ]; then
            echo "📁 $backup_type:"
            ls -lt "$backup_path" | head -5 | sed 's/^/   /'
            echo
        fi
    done
    
    # Показываем snapshots
    if [ -d "$BACKUP_ROOT/daily_snapshots" ]; then
        echo "📸 Daily snapshots:"
        ls -lt "$BACKUP_ROOT/daily_snapshots" | grep "$stage" | head -5 | sed 's/^/   /'
        echo
    fi
    
    read -p "Введите полный путь к backup файлу: " backup_file
    
    if [ -f "$backup_file" ]; then
        local filename=$(basename "$backup_file")
        local target_file=""
        
        # Определяем целевой файл по имени backup'а
        if [[ "$filename" == *"improve_baseline_model"* ]]; then
            target_file="improve_baseline_model.ipynb"
        elif [[ "$filename" == *"register_model_mlflow"* ]]; then
            target_file="register_model_mlflow.py"
        elif [[ "$filename" == *"improve_dataset"* ]]; then
            target_file="improve_dataset.py"
        else
            echo "Доступные файлы в текущем этапе:"
            get_files_to_protect "$stage"
            read -p "Введите имя целевого файла: " target_file
        fi
        
        if [ -n "$target_file" ]; then
            cp "$backup_file" "$target_file"
            print_success "Файл восстановлен: $target_file ← $filename"
            
            # Переходим в корень для Git операций
            cd "$PROJECT_ROOT"
            
            # Коммитим восстановление
            git add "$target_file"
            git commit -m "[$stage] Emergency restore: $target_file from $filename"
            print_success "Восстановление зафиксировано в Git"
            
            # Возвращаемся в рабочую директорию
            cd - > /dev/null
        fi
    else
        print_error "Backup файл не найден: $backup_file"
    fi
}

# 6. ОЧИСТКА СТАРЫХ BACKUP'ОВ
cleanup_old_backups() {
    print_header "ОЧИСТКА СТАРЫХ BACKUP'ОВ"
    
    echo "🧹 Очищаю backup'ы старше 7 дней..."
    local total_deleted=0
    
    # Очищаем backup'ы во всех этапах
    for stage_dir in "$BACKUP_ROOT"/{stage1,stage2-5,root}; do
        if [ -d "$stage_dir" ]; then
            local stage_name=$(basename "$stage_dir")
            echo "📁 Этап: $stage_name"
            
            # Очищаем каждый тип backup'ов
            for backup_type in auto manual pre_commit experiments; do
                local backup_path="$stage_dir/$backup_type"
                if [ -d "$backup_path" ]; then
                    local deleted=$(find "$backup_path" -type f -mtime +7 -delete -print 2>/dev/null | wc -l)
                    total_deleted=$((total_deleted + deleted))
                    if [ "$deleted" -gt 0 ]; then
                        print_success "  $backup_type: удалено $deleted файлов"
                    fi
                fi
            done
        fi
    done
    
    # Очищаем daily snapshots старше 2 дней
    if [ -d "$BACKUP_ROOT/daily_snapshots" ]; then
        local deleted_snapshots=$(find "$BACKUP_ROOT/daily_snapshots" -type f -mtime +2 -delete -print 2>/dev/null | wc -l)
        total_deleted=$((total_deleted + deleted_snapshots))
        if [ "$deleted_snapshots" -gt 0 ]; then
            print_success "Snapshots: удалено $deleted_snapshots файлов"
        fi
    fi
    
    # Статистика после очистки
    local remaining_total=$(find "$BACKUP_ROOT" -type f 2>/dev/null | wc -l)
    local backup_size=$(du -sh "$BACKUP_ROOT" 2>/dev/null | cut -f1)
    
    print_success "Всего удалено: $total_deleted файлов"
    print_success "Осталось backup'ов: $remaining_total"
    print_success "Размер backup'ов: $backup_size"
}

# ГЛАВНОЕ МЕНЮ
show_menu() {
    print_header "СИСТЕМА БЕЗОПАСНОСТИ КОДА"
    echo "1) 🌅 Утренняя настройка"
    echo "2) 💾 Ежечасный backup"
    echo "3) 🧪 Подготовка к эксперименту"
    echo "4) 🌙 Вечернее сохранение"
    echo "5) 🚨 Экстренное восстановление"
    echo "6) 🧹 Очистка старых backup'ов"
    echo "7) 📊 Статистика проекта"
    echo "0) Выход"
    echo
}

# СТАТИСТИКА ПРОЕКТА
show_stats() {
    print_header "СТАТИСТИКА МНОГОЭТАПНОГО ПРОЕКТА"
    
    # Определяем текущий этап
    if ! project_safety_check; then
        print_error "Ошибка определения этапа проекта"
        return 1
    fi
    
    local stage="$PROJECT_STAGE"
    echo
    
    print_success "Текущий этап: $(get_stage_description $stage)"
    echo
    
    # Переходим в корень для Git статистики
    cd "$PROJECT_ROOT"
    
    # Общая статистика проекта
    local total_commits=$(git log --oneline | wc -l)
    local project_size=$(du -sh . | cut -f1)
    local git_branches=$(git branch -a | wc -l)
    
    echo "🚀 ОБЩАЯ СТАТИСТИКА ПРОЕКТА:"
    echo "   📝 Git коммитов: $total_commits"
    echo "   📂 Размер проекта: $project_size"
    echo "   🌿 Git веток: $git_branches"
    echo
    
    # Статистика backup'ов по этапам
    echo "💾 СТАТИСТИКА BACKUP'ОВ:"
    for stage_name in stage1 stage2-5 root; do
        local stage_path="$BACKUP_ROOT/$stage_name"
        if [ -d "$stage_path" ]; then
            local stage_backups=$(find "$stage_path" -type f 2>/dev/null | wc -l)
            local stage_size=$(du -sh "$stage_path" 2>/dev/null | cut -f1)
            echo "   📁 $stage_name: $stage_backups файлов ($stage_size)"
        fi
    done
    
    # Статистика daily snapshots
    if [ -d "$BACKUP_ROOT/daily_snapshots" ]; then
        local snapshots_count=$(find "$BACKUP_ROOT/daily_snapshots" -type f | wc -l)
        local snapshots_size=$(du -sh "$BACKUP_ROOT/daily_snapshots" | cut -f1)
        echo "   📸 Snapshots: $snapshots_count файлов ($snapshots_size)"
    fi
    
    echo
    
    # Статистика текущего этапа
    echo "📊 СТАТИСТИКА ТЕКУЩЕГО ЭТАПА ($stage):"
    
    # Возвращаемся в рабочую директорию
    cd - > /dev/null
    
    local files_to_protect=($(get_files_to_protect "$stage"))
    echo "   📋 Файлов в этапе: ${#files_to_protect[@]}"
    
    for file in "${files_to_protect[@]}"; do
        if [ -f "$file" ]; then
            local file_size=$(du -h "$file" | cut -f1)
            local last_modified=$(stat -c %y "$file" | cut -d' ' -f1,2 | cut -d'.' -f1)
            echo "   📄 $file: $file_size (изменен: $last_modified)"
        fi
    done
    
    echo
    echo "🕐 Последняя проверка: $(date '+%Y-%m-%d %H:%M:%S')"
}

# Главный цикл программы
main() {
    while true; do
        show_menu
        read -p "Выберите действие: " choice
        
        case $choice in
            1) morning_setup ;;
            2) hourly_backup ;;
            3) safe_experiment ;;
            4) evening_save ;;
            5) emergency_restore ;;
            6) cleanup_old_backups ;;
            7) show_stats ;;
            0) 
                print_success "До свидания! Код в безопасности 🛡️"
                exit 0
                ;;
            *)
                print_error "Неверный выбор: $choice"
                ;;
        esac
        
        echo
        read -p "Нажмите Enter для продолжения..."
        clear
    done
}

# Запуск программы
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    clear
    main "$@"
fi
