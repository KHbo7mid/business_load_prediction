import logging
from pathlib import Path

import pandas as pd

from pipeline.transform.common import (
    read_jsonl_files,
    object_id_to_str,
    mongo_date_to_value,
    prepare_parquet_dataset_dir,
    write_parquet_file_from_chunks,
)
from configs.config import load_config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

RAW = Path("data/raw/performance_logs")
OUT = Path("data/staging/performance_logs_clean")
config=load_config()
MIN_DATE = pd.Timestamp(f"{config.EXPORT_DATE_MIN}", tz="UTC")
MAX_DATE = pd.Timestamp(f"{config.EXPORT_DATE_MAX}", tz="UTC")

# Response time is measured in milliseconds.
# Negative response time is physically impossible, so those rows are invalid.
MIN_RESPONSE_TIME_MS = 0

# Very large API durations usually come from instrumentation problems,
# stuck requests, or log lifecycle artifacts.
# We cap them so one broken log row does not dominate hourly p95/mean metrics.
MAX_RESPONSE_TIME_MS = 120_000


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df["createdAt"] = df["createdAt"].apply(mongo_date_to_value)
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    df = df.dropna(subset=["createdAt"])
    df = df[(df["createdAt"] >= MIN_DATE) & (df["createdAt"] <= MAX_DATE)]

    df["log_id"] = df["_id"].apply(object_id_to_str)
    df["endpoint"] = df["endpoint"].astype("string").str.strip()
    df["method"] = df["method"].astype("string").str.upper().str.strip()
    df["statusCode"] = pd.to_numeric(df["statusCode"], errors="coerce")
    df["responseTime"] = pd.to_numeric(df["responseTime"], errors="coerce")

    df = df.dropna(subset=["endpoint", "method", "responseTime"])
    df = df[~df["endpoint"].str.lower().isin(["", "none", "null", "nan"])]
    df = df[~df["method"].str.lower().isin(["", "none", "null", "nan"])]

    invalid_response_time_count = int((df["responseTime"] < MIN_RESPONSE_TIME_MS).sum())
    capped_response_time_count = int((df["responseTime"] > MAX_RESPONSE_TIME_MS).sum())

    df = df[df["responseTime"] >= MIN_RESPONSE_TIME_MS]
    df["responseTime"] = df["responseTime"].clip(upper=MAX_RESPONSE_TIME_MS)

    result = df[["log_id", "createdAt", "endpoint", "method", "statusCode", "responseTime"]]

    log.info(
        "performance logs chunk: %s -> %s rows; invalid_response_time=%s; capped_response_time=%s",
        before,
        len(result),
        invalid_response_time_count,
        capped_response_time_count,
    )
    return result


def main():
    prepare_parquet_dataset_dir(OUT)

    total_rows = 0
    output_file_count = 0
    for raw_file_path, raw_chunks in read_jsonl_files(RAW, pattern="performance_logs_*.jsonl"):
        output_file_path = OUT / f"{raw_file_path.stem.replace('performance_logs_', 'performance_logs_clean_', 1)}.parquet"
        cleaned_chunks = (clean_chunk(chunk) for chunk in raw_chunks)
        cleaned_rows = write_parquet_file_from_chunks(cleaned_chunks, output_file_path)
        total_rows += cleaned_rows
        output_file_count += 1
        log.info("saved %s rows to %s", cleaned_rows, output_file_path)

    log.info("saved %s rows across %s yearly parquet files to %s", total_rows, output_file_count, OUT)


if __name__ == "__main__":
    main()
