#!/usr/bin/env bash
set -euo pipefail 

source .env

mkdir -p data/raw/orders

mongoexport \
    --uri="$MONGO_URL" \
    --db="$MONGO_DB" \
    --collection="orders" \
    --fields="_id,createdAt,orderStatus,destination.address.city,destination.address.area,destination.address.street,destination.address.location,destination.duration,destination.distance,destination.eta,branch,merchant,amount,deliveryFee,driverSearchTrials,country,platformName,deleted" \
    --query="{
     \"createdAt\": {
      \"\$gte\": {\"\$date\": \"${EXPORT_DATE_MIN}T00:00:00.000Z\"},
      \"\$lte\": {\"\$date\": \"${EXPORT_DATE_MAX}T23:59:59.999Z\"}
    },
    \"orderStatus\": {\"\$ne\": null}
    }" \
    --type=json \
    --out="data/raw/orders/orders_${EXPORT_DATE_MIN}_${EXPORT_DATE_MAX}.jsonl"


wc -l data/raw/orders/orders_${EXPORT_DATE_MIN}_${EXPORT_DATE_MAX}.jsonl
