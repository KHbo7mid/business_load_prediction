import logging
from pathlib import Path
import pandas as pd
from configs.config import load_config
from pipeline.transform.common import (
    read_jsonl_files,
    object_id_to_str,
    mongo_date_to_value,
    prepare_parquet_dataset_dir,
    write_parquet_file_from_chunks,
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log=logging.getLogger(__name__)

RAW=Path("data/raw/delivery_requests")
OUTPUT=Path("data/staging/delivery_requests_clean")
config = load_config()
MIN_DATE = pd.Timestamp(f"{config.EXPORT_DATE_MIN}", tz="UTC")
MAX_DATE = pd.Timestamp(f"{config.EXPORT_DATE_MAX}", tz="UTC")

def clean_chunk(df:pd.DataFrame) -> pd.DataFrame:
    before =len(df)
    
    df["createdAt"]=df["createdAt"].apply(mongo_date_to_value)
    df["createdAt"]=pd.to_datetime(df["createdAt"],errors="coerce",utc=True)
    df=df.dropna(subset=["createdAt"])
    df=df[(df["createdAt"] >=MIN_DATE) & (df["createdAt"] <=MAX_DATE)]
    
    df["status"]=df["status"].astype(str).str.lower().str.strip()
    df = df[~df["status"].isin(["", "none", "null", "nan"])]
    
    df["request_id"]=df["_id"].apply(object_id_to_str)
    df["order_id"]=df["order"].apply(object_id_to_str)
    df["driver_id"]=df["driver"].apply(object_id_to_str)
    
    result =df[["request_id","order_id","driver_id","status","createdAt"]]
    log.info(f"Cleaned chunk: {before} -> {len(result)} rows")
    return result

def main():
    prepare_parquet_dataset_dir(OUTPUT)
    total_rows = 0
    output_file_count = 0
    for raw_file_path, raw_chunks in read_jsonl_files(RAW, pattern="delivery_requests_*.jsonl", chunk_size=200_000):
        output_file_path = OUTPUT / f"{raw_file_path.stem.replace('delivery_requests_', 'delivery_requests_clean_', 1)}.parquet"
        cleaned_chunks = (clean_chunk(chunk) for chunk in raw_chunks)
        cleaned_rows = write_parquet_file_from_chunks(cleaned_chunks, output_file_path)
        total_rows += cleaned_rows
        output_file_count += 1
        log.info("saved %s rows to %s", cleaned_rows, output_file_path)
    log.info("saved %s rows across %s yearly parquet files to %s", total_rows, output_file_count, OUTPUT)
    
if __name__ == "__main__":
    main()
