#!/usr/bin/env bash
set -euo pipefail

source .env

mkdir -p data/raw/performance_logs

mongoexport \
  --uri="$MONGO_URL" \
  --db="$MONGO_DB" \
  --collection="performancelogs" \
  --fields="_id,createdAt,endpoint,method,statusCode,responseTime" \
  --query="{
    \"createdAt\": {
      \"\$gte\": {\"\$date\": \"${EXPORT_DATE_MIN}T00:00:00.000Z\"},
      \"\$lte\": {\"\$date\": \"${EXPORT_DATE_MAX}T23:59:59.999Z\"}
    }
  }" \
  --type=json \
  --out="data/raw/performance_logs/performance_logs_${EXPORT_DATE_MIN}_${EXPORT_DATE_MAX}.jsonl"

wc -l data/raw/performance_logs/performance_logs_${EXPORT_DATE_MIN}_${EXPORT_DATE_MAX}.jsonl
