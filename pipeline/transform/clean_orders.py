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
from pipeline.transform.area_normalization import normalize_area_frame


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log=logging.getLogger(__name__)

RAW=Path("data/raw/orders")   
OUTPUT=Path("data/staging/orders_clean")

config = load_config()
MIN_DATE = pd.Timestamp(f"{config.EXPORT_DATE_MIN}", tz="UTC")
MAX_DATE = pd.Timestamp(f"{config.EXPORT_DATE_MAX}", tz="UTC")

def extract_address_field(destination,field):
    if not isinstance(destination,dict):
        return None
    address=destination.get("address")
    if not isinstance(address,dict):
        return None
    return address.get(field)


def extract_location_field(destination, field):
    if not isinstance(destination, dict):
        return None
    address = destination.get("address")
    if not isinstance(address, dict):
        return None
    location = address.get("location")
    if not isinstance(location, dict):
        return None
    return location.get(field)


def clean_chunk(df:pd.DataFrame) -> pd.DataFrame:
    before =len(df)
    
    deleted_text=df.get("deleted",pd.Series([None] *len(df))).astype(str).str.lower().str.strip()
    df = df[~deleted_text.isin(["true","1","yes"])]
    
    df["createdAt"]=df["createdAt"].apply(mongo_date_to_value)
    df["createdAt"]=pd.to_datetime(df["createdAt"],errors="coerce",utc=True)
    df=df.dropna(subset=["createdAt"])
    df=df[(df["createdAt"] >=MIN_DATE) & (df["createdAt"] <=MAX_DATE)]
    
    df["orderStatus"]=df["orderStatus"].astype(str).str.lower().str.strip()
    df = df[~df["orderStatus"].isin(["", "none", "null", "nan"])]
    
    df["order_id"] = df["_id"].apply(object_id_to_str)
    df["branch_id"] = df["branch"].apply(object_id_to_str)
    df["merchant_id"] = df["merchant"].apply(object_id_to_str)
    
    df["area_raw"]=df["destination"].apply(lambda x: extract_address_field(x,"area"))
    df["city"]=df["destination"].apply(lambda x: extract_address_field(x,"city"))
    df["street"]=df["destination"].apply(lambda x: extract_address_field(x,"street"))
    df["latitude"] = df["destination"].apply(lambda x: extract_location_field(x, "latitude"))
    df["longitude"] = df["destination"].apply(lambda x: extract_location_field(x, "longitude"))
    area_fields = normalize_area_frame(df["city"], df["area_raw"])
    df[["area", "area_original", "area_quality_flag", "sub_area"]] = area_fields

    for col in  [
        "latitude",
        "longitude",
    ]:
        df[col] = pd.to_numeric(df[col],errors="coerce")
    keep=[
        "order_id",
        "branch_id",
        "merchant_id",
        "createdAt",
        "orderStatus",
        "area",
        "area_original",
        "area_quality_flag",
        "sub_area",
        "city",
        "area_raw",
        "latitude",
        "longitude",
        "country",
        "platformName"
    ]
    result =df[[c for c in keep if c in df.columns]]
    log.info(f"Cleaned chunk: {before} -> {len(result)} rows")
    return result

def main():
    prepare_parquet_dataset_dir(OUTPUT)
    
    total_rows = 0
    output_file_count = 0
    for raw_file_path, raw_chunks in read_jsonl_files(RAW, pattern="orders_*.jsonl"):
        output_file_path = OUTPUT / f"{raw_file_path.stem.replace('orders_', 'orders_clean_', 1)}.parquet"
        cleaned_chunks = (clean_chunk(chunk) for chunk in raw_chunks)
        cleaned_rows = write_parquet_file_from_chunks(cleaned_chunks, output_file_path)
        total_rows += cleaned_rows
        output_file_count += 1
        log.info("saved %s rows to %s", cleaned_rows, output_file_path)
        
    log.info("saved %s rows across %s yearly parquet files to %s", total_rows, output_file_count, OUTPUT)
    
if __name__ == "__main__":
    main()
