import json 
import shutil
from pathlib import Path 
from typing import Iterator
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def iter_jsonl_paths(path: Path, pattern: str = "*.jsonl") -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob(pattern))
    return []


def read_jsonl(path:Path,chunk_size:int=100_000)-> Iterator[pd.DataFrame]:
    buffer =[]
    
    with path.open('r',encoding="utf-8") as f:
        for line in f:
            line= line.strip()
            if not line :
                continue
            
            try:
                buffer.append(json.loads(line))
            except json.JSONDecodeError :
                continue
            
            if len(buffer) >= chunk_size:
                yield pd.DataFrame(buffer)
                buffer= []
    if buffer:
        yield pd.DataFrame(buffer)


def read_jsonl_many(path: Path, pattern: str = "*.jsonl", chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
    paths = iter_jsonl_paths(path, pattern=pattern)
    if not paths:
        raise FileNotFoundError(f"No JSONL files found at {path} with pattern {pattern}")

    for file_path in paths:
        yield from read_jsonl(file_path, chunk_size=chunk_size)


def read_jsonl_files(path: Path, pattern: str = "*.jsonl", chunk_size: int = 100_000) -> Iterator[tuple[Path, Iterator[pd.DataFrame]]]:
    paths = iter_jsonl_paths(path, pattern=pattern)
    if not paths:
        raise FileNotFoundError(f"No JSONL files found at {path} with pattern {pattern}")

    for file_path in paths:
        yield file_path, read_jsonl(file_path, chunk_size=chunk_size)


def prepare_parquet_dataset_dir(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def write_parquet_file_from_chunks(chunks: Iterator[pd.DataFrame], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    try:
        for chunk in chunks:
            if chunk.empty:
                continue
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    return total_rows
        
def object_id_to_str(value):
    if isinstance(value,dict) and '$oid' in value:
        return value['$oid']
    if value is None or pd.isna(value):
        return None
    return str(value)


def mongo_date_to_value(value):
    if isinstance(value,dict) and "$date" in value:
        return value["$date"]
    return value

def normalize_text(value):
    if value is None or pd.isna(value):
        return None
    value =str(value).strip()
    if not value:
        return None
    return " ".join(value.split()).title()

def is_numeric_text(value):
    if value is None:
        return False

    return str(value).strip().isdigit()
