#!/usr/bin/env bash

# Загрузка переменных окружения
if [ -f "../.env" ]; then
    while IFS='=' read -r key value; do
        case "$key" in
            '#'*) ;;
            '') ;;
            *)
                value="${value%\"}"
                value="${value#\"}"
                export "$key"="$value"
        esac
    done < "../.env"
fi

# Экспорт переменных для MLflow
export MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"

# Освобождение порта 5000
PORT=5000
if lsof -i :$PORT &>/dev/null; then
    lsof -t -i :$PORT | xargs -r kill -9
    sleep 2
fi

# Запуск MLflow сервера
MLFLOW_SERVER_HOST=${TRACKING_SERVER_HOST:-127.0.0.1}
MLFLOW_SERVER_PORT=${TRACKING_SERVER_PORT:-5000}

exec mlflow server \
  --backend-store-uri "postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME" \
  --default-artifact-root "s3://$S3_BUCKET_NAME" \
  --host $MLFLOW_SERVER_HOST \
  --port $MLFLOW_SERVER_PORT