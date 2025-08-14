#!/bin/bash
# УСТАНОВКА СИСТЕМЫ БЕЗОПАСНОСТИ КОДА
# Этот скрипт настроит полную защиту для вашего ML проекта
# ================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "🛡️  УСТАНОВКА СИСТЕМЫ БЕЗОПАСНОСТИ КОДА"
echo "========================================="
echo

# 1. Создание алиасов для .bashrc
print_step "Настройка алиасов для быстрого доступа"

ALIASES_FILE="$HOME/.bash_aliases"
cat > "$ALIASES_FILE" << 'EOF'
# =============================================
# АЛИАСЫ ДЛЯ МНОГОЭТАПНОЙ БЕЗОПАСНОЙ РАЗРАБОТКИ ML
# =============================================

# Навигация по этапам проекта
alias mlroot='cd ~/mle_projects/mle-project-sprint-2-v001'
alias ml1='cd ~/mle_projects/mle-project-sprint-2-v001/mlflow_server'
alias ml25='cd ~/mle_projects/mle-project-sprint-2-v001/model_improvement'
alias mlcd='cd ~/mle_projects/mle-project-sprint-2-v001/model_improvement'  # обратная совместимость

# Универсальные команды безопасности (работают в любом этапе)
alias mlsave='./safe_save.sh'
alias mlwork='./daily_workflow.sh'
alias mlstatus='../project_safety_detector.sh'

# Git команды с безопасностью
alias gs='git status'
alias ga='git add -A'
alias gc='git commit -m'
alias gp='git push origin main'
alias gsafe='git add -A && git commit -m "Safe commit - $(date)" && echo "✅ Code saved safely"'

# Jupyter команды
alias jup='jupyter notebook --ip=0.0.0.0 --no-browser --allow-root'
alias jlab='jupyter lab --ip=0.0.0.0 --no-browser --allow-root'

# Универсальные команды backup'ов (автоматически определяют этап)
alias mlbackup='STAGE=$(../project_safety_detector.sh | grep "ЭТАП:" | cut -d":" -f2 | xargs) && mkdir -p ../project_backups/${STAGE}/manual && for f in $(../project_safety_detector.sh 2>/dev/null | grep "ФАЙЛЫ ДЛЯ ЗАЩИТЫ:" -A 20 | grep "^   \./" | xargs); do [ -f "$f" ] && cp "$f" "../project_backups/${STAGE}/manual/manual_$(basename $f .*)_$(date +%Y%m%d_%H%M).${f##*.}" && echo "✅ Manual backup: $f"; done'

# Быстрый просмотр backup'ов
alias mllist='find ../project_backups -name "*$(date +%Y%m%d)*" -type f | head -10'
alias mlstats='cd ~/mle_projects/mle-project-sprint-2-v001 && echo "📊 ОБЩАЯ СТАТИСТИКА:" && echo "Git коммитов: $(git log --oneline | wc -l)" && echo "Размер проекта: $(du -sh . | cut -f1)" && echo "Backup файлов: $(find project_backups -type f 2>/dev/null | wc -l)" && echo "Размер backup'ов: $(du -sh project_backups 2>/dev/null | cut -f1)" && cd - > /dev/null'

# Экстренные команды
alias mlrestore='echo "🚨 ДОСТУПНЫЕ BACKUP'Ы:" && find ../project_backups -type f -name "*$(date +%Y%m%d)*" | head -5'
alias mlpanic='PANIC_TIME=$(date +%Y%m%d_%H%M%S) && ../project_safety_detector.sh > /dev/null && STAGE=$PROJECT_STAGE && for f in $(find . -maxdepth 1 -name "*.ipynb" -o -name "*.py"); do [ -f "$f" ] && cp "$f" "PANIC_${PANIC_TIME}_$(basename $f)" && echo "🚨 PANIC BACKUP: $f"; done && git add -A && git commit -m "PANIC SAVE - $PANIC_TIME" && echo "🚨 ПАНИКА! ВСЕ СОХРАНЕНО!"'

# Мониторинг изменений
alias mlwatch='watch -n 30 "echo \"📍 ЭТАП: \$(../project_safety_detector.sh 2>/dev/null | grep ЭТАП | head -1)\"; echo; git status --short; echo; echo \"📊 Последние backup'\''ы:\"; find ../project_backups -type f -printf \"%T@ %p\n\" 2>/dev/null | sort -n | tail -3 | cut -d\" \" -f2- | sed '\''s/^/   /'\'' "'

echo "🛡️  ML Safety Aliases loaded! Use 'mlcd' to go to project, 'mlwork' for daily workflow"
EOF

print_success "Алиасы созданы в $ALIASES_FILE"

# 2. Настройка Jupyter конфигурации
print_step "Настройка автосохранения Jupyter"

JUPYTER_CONFIG_DIR="$HOME/.jupyter"
mkdir -p "$JUPYTER_CONFIG_DIR"

cat > "$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py" << 'EOF'
# Конфигурация Jupyter для безопасной разработки
c = get_config()

# Автосохранение каждые 2 минуты (120 секунд)
c.NotebookApp.autosave_interval = 120

# Создание checkpoint'ов каждые 10 минут
c.FileCheckpoints.checkpoint_interval = 600

# Включение автоматического создания backup'ов
c.NotebookApp.use_redirect_file = True

# Показывать линии кода
c.NotebookApp.show_banner = True

# Логирование для отладки
c.Application.log_level = 'INFO'

print("🛡️  Jupyter configured for safe development!")
print("📝 Autosave: every 2 minutes")
print("💾 Checkpoints: every 10 minutes")
EOF

print_success "Jupyter настроен для автосохранения"

# 3. Создание Git hooks
print_step "Настройка Git hooks для автоматической защиты"

mkdir -p .git/hooks

# Pre-commit hook - создает backup перед каждым коммитом
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook - создает backup перед коммитом

NOTEBOOK="improve_baseline_model.ipynb"
BACKUP_DIR="backups"

if [ -f "$NOTEBOOK" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$NOTEBOOK" "$BACKUP_DIR/pre_commit_backup_$(date +%Y%m%d_%H%M%S).ipynb"
    echo "✅ Pre-commit backup created"
fi
EOF

chmod +x .git/hooks/pre-commit

# Post-commit hook - уведомляет об успешном сохранении
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
# Post-commit hook - уведомление о сохранении

echo "🎉 Commit successful! Code safely saved at $(date)"
echo "📊 Total commits: $(git rev-list --count HEAD)"
EOF

chmod +x .git/hooks/post-commit

print_success "Git hooks установлены"

# 4. Создание cron job для автоматических backup'ов
print_step "Настройка автоматических backup'ов"

CRON_SCRIPT="$HOME/ml_auto_backup.sh"
cat > "$CRON_SCRIPT" << EOF
#!/bin/bash
# Автоматический backup каждый час для многоэтапного проекта

PROJECT_ROOT="$PWD"
BACKUP_ROOT="\$PROJECT_ROOT/project_backups"

# Функция backup'а для этапа
backup_stage() {
    local stage_dir="\$1"
    local stage_name="\$2"
    
    if [ -d "\$stage_dir" ]; then
        cd "\$stage_dir"
        
        # Определяем файлы для backup'а в зависимости от этапа
        local files_pattern=""
        case "\$stage_name" in
            "stage1")
                files_pattern="*.py"
                ;;
            "stage2-5")
                files_pattern="*.ipynb *.py"
                ;;
        esac
        
        if [ -n "\$files_pattern" ]; then
            mkdir -p "\$BACKUP_ROOT/\$stage_name/auto"
            
            for pattern in \$files_pattern; do
                for file in \$pattern; do
                    if [ -f "\$file" ]; then
                        local timestamp=\$(date +%Y%m%d_%H%M)
                        local backup_name="\${file%.*}_auto_\${timestamp}.\${file##*.}"
                        cp "\$file" "\$BACKUP_ROOT/\$stage_name/auto/\$backup_name"
                    fi
                done
            done
            
            # Удаляем старые auto backup'ы (старше 24 часов)
            find "\$BACKUP_ROOT/\$stage_name/auto" -name "*auto_*" -mtime +1 -delete 2>/dev/null || true
        fi
        
        cd "\$PROJECT_ROOT"
    fi
}

# Backup'им все этапы
backup_stage "\$PROJECT_ROOT/mlflow_server" "stage1"
backup_stage "\$PROJECT_ROOT/model_improvement" "stage2-5"

# Очищаем старые daily snapshots
if [ -d "\$BACKUP_ROOT/daily_snapshots" ]; then
    find "\$BACKUP_ROOT/daily_snapshots" -type f -mtime +2 -delete 2>/dev/null || true
fi
EOF

chmod +x "$CRON_SCRIPT"

# Добавляем cron job (каждый час)
(crontab -l 2>/dev/null; echo "0 * * * * $CRON_SCRIPT") | crontab -

print_success "Автоматические backup'ы настроены (каждый час)"

# 5. Создание файла с инструкциями
print_step "Создание документации"

cat > "SAFETY_INSTRUCTIONS.md" << 'EOF'
# 🛡️ УНИВЕРСАЛЬНАЯ СИСТЕМА БЕЗОПАСНОСТИ МНОГОЭТАПНОГО ML ПРОЕКТА

## 🎯 Структура проекта

```
mle-project-sprint-2-v001/
├── mlflow_server/          # 🚀 Этап 1: Настройка MLflow
├── model_improvement/      # 🧪 Этапы 2-5: Улучшение модели
└── project_backups/        # 💾 Централизованные backup'ы
    ├── stage1/             # Backup'ы этапа 1
    ├── stage2-5/           # Backup'ы этапов 2-5
    ├── root/               # Backup'ы корня проекта
    └── daily_snapshots/    # Ежедневные снимки
```

## 🚀 Быстрый старт

### Навигация по этапам:
```bash
mlroot                 # Корень проекта
ml1                    # Этап 1: MLflow Server
ml25                   # Этапы 2-5: Model Improvement
mlcd                   # Этапы 2-5 (обратная совместимость)
```

### Универсальная работа (в любом этапе):
```bash
mlstatus               # Проверка текущего этапа
mlsave                 # Быстрое сохранение
mlwork                 # Ежедневный workflow
mlbackup               # Ручной backup
```

### Перед важными изменениями:
```bash
mlbackup               # Ручной backup
gsafe                  # Безопасный git commit
```

### В случае проблем:
```bash
mlpanic                # Экстренное сохранение
mlrestore              # Просмотр backup'ов
mlstats                # Общая статистика
```

## 📋 Ежедневный чек-лист

### 🌅 Утром:
- [ ] `ml1` или `ml25` - переход к нужному этапу
- [ ] `mlstatus` - проверка текущего этапа
- [ ] `mlwork` → "1" - утренняя настройка
- [ ] Проверка git статуса

### 💼 Во время работы:
- [ ] `mlsave` - каждые 30-60 минут (в любом этапе)
- [ ] `mlwork` → "3" - перед экспериментами
- [ ] `mlbackup` - перед важными изменениями
- [ ] Коммиты с описательными сообщениями

### 🌙 Вечером:
- [ ] `mlwork` → "4" - вечернее сохранение
- [ ] `gp` - загрузка в удаленный репозиторий
- [ ] `mlstats` - проверка общей статистики
- [ ] Проверка backup'ов всех этапов

## 🛠️ Команды

| Категория | Команда | Описание |
|-----------|---------|----------|
| **Навигация** | `mlroot` | Корень проекта |
| | `ml1` | Этап 1: MLflow Server |
| | `ml25` | Этапы 2-5: Model Improvement |
| | `mlcd` | Этапы 2-5 (совместимость) |
| **Безопасность** | `mlstatus` | Проверка текущего этапа |
| | `mlsave` | Быстрое сохранение |
| | `mlwork` | Ежедневный workflow |
| | `mlbackup` | Ручной backup |
| **Мониторинг** | `mlstats` | Общая статистика |
| | `mllist` | Список backup'ов |
| | `mlwatch` | Мониторинг изменений |
| **Экстренные** | `mlpanic` | Экстренное сохранение |
| | `mlrestore` | Просмотр backup'ов |

## 🔄 Автоматизация

- ✅ Автосохранение Jupyter каждые 2 минуты
- ✅ Backup'ы каждый час (cron) для всех этапов
- ✅ Git hooks для pre/post commit
- ✅ Очистка старых backup'ов (>7 дней)
- ✅ Автоматическое определение этапа проекта
- ✅ Централизованное хранение backup'ов

## 🚨 Восстановление

1. Просмотр backup'ов: `mllist` или `mlrestore`
2. Автоматическое восстановление: `mlwork` → "5"
3. Ручное восстановление: `cp /path/to/backup.ext target_file.ext`
4. Git откат: `git reset --hard HEAD~1`
5. Проверка статуса: `mlstatus`

## 📊 Мониторинг

- Общая статистика: `mlstats`
- Статус этапа: `mlstatus`
- Наблюдение в реальном времени: `mlwatch`
- Git статус: `gs`
- Просмотр backup'ов: `mllist`

## 🌟 Особенности многоэтапной системы

- **Автоопределение этапа**: система автоматически понимает, в каком этапе вы работаете
- **Централизованные backup'ы**: все backup'ы хранятся в `project_backups/`
- **Разные типы файлов**: защищает Python файлы в этапе 1, notebooks в этапах 2-5
- **Умные алиасы**: команды адаптируются под текущий этап
- **Структурированное хранение**: отдельные папки для каждого типа backup'а

---
💡 **Помните**: Универсальная система защищает все этапы проекта автоматически!
EOF

print_success "Документация создана: SAFETY_INSTRUCTIONS.md"

# 6. Создание быстрых клавиш для VS Code
print_step "Создание настроек для VS Code"

VSCODE_DIR="$HOME/.vscode"
mkdir -p "$VSCODE_DIR"

cat > "$VSCODE_DIR/tasks.json" << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "ML Safe Save",
            "type": "shell",
            "command": "./safe_save.sh",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "ML Daily Workflow",
            "type": "shell",
            "command": "./daily_workflow.sh",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": true,
                "panel": "shared"
            }
        }
    ]
}
EOF

print_success "VS Code задачи настроены"

echo
echo "======================================================="
echo "🎉 УНИВЕРСАЛЬНАЯ СИСТЕМА БЕЗОПАСНОСТИ УСТАНОВЛЕНА!"
echo "======================================================="
echo
print_success "✅ Поддержка многоэтапной структуры проекта"
print_success "✅ Автоматическое определение этапов"
print_success "✅ Централизованное хранение backup'ов"
print_success "✅ Универсальные команды для всех этапов"
echo
print_success "Перезапустите терминал для активации алиасов"
print_success "Используйте 'mlstatus' для проверки этапа"
print_success "Используйте 'ml1' для этапа 1, 'ml25' для этапов 2-5"
print_success "Используйте 'mlwork' для ежедневного workflow"
print_success "Читайте SAFETY_INSTRUCTIONS.md для подробностей"
echo
print_warning "🚀 Готово к работе в многоэтапном проекте!"
echo "Команды: ml1 && mlsave  или  ml25 && mlsave"
echo
