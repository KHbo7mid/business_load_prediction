#!/usr/bin/env bash
set -euo pipefail

source .env

mkdir -p data/raw/delivery_requests

mongoexport \
    --uri="$MONGO_URL" \
    --db="$MONGO_DB" \
    --collection="deliveryrequests" \
    --fields="_id,createdAt,order,driver,status" \
    --query="{
     \"createdAt\": {
      \"\$gte\": {\"\$date\": \"${EXPORT_DATE_MIN}T00:00:00.000Z\"},
      \"\$lte\": {\"\$date\": \"${EXPORT_DATE_MAX}T23:59:59.999Z\"}
    }
    }" \
    --type=json \
    --out="data/raw/delivery_requests/delivery_requests_${EXPORT_DATE_MIN}_${EXPORT_DATE_MAX}.jsonl"

wc -l data/raw/delivery_requests/delivery_requests_${EXPORT_DATE_MIN}_${EXPORT_DATE_MAX}.jsonl
