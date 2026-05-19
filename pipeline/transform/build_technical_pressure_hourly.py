import logging 
from pathlib import Path
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log=logging.getLogger(__name__)

PERF = Path("data/staging/performance_logs_clean")
OUT = Path("data/analytical/technical_pressure_hourly.parquet")

def normalize_endpoint(endpoint:str) -> str:
    endpoint=str(endpoint).strip().lower()
    return endpoint

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df=pd.read_parquet(PERF)
    df["window_start"]=df["createdAt"].dt.floor("h")
    df["endpoint"]=df["endpoint"].apply(normalize_endpoint)
    df["is_error"] = (df["statusCode"] >= 500).astype("int8")
    
    group_cols = ["window_start","endpoint","method"]
    grouped = df.groupby(group_cols, sort=False)
    result=(
        grouped
        .agg(
            request_count=("log_id","count"),
            avg_response_time=("responseTime", "mean"),
            error_count=("is_error", "sum"),
        )
        .reset_index()
    )
    p95_response_time = (
        grouped["responseTime"]
        .quantile(0.95)
        .rename("p95_response_time")
        .reset_index()
    )
    result = result.merge(p95_response_time, on=group_cols, how="left")
    
    result["error_rate"] = result["error_count"] / result["request_count"]
    result.to_parquet(OUT, index=False)
    
    log.info("saved technical pressure: %s rows", len(result))
    
    
if __name__ == "__main__":
    main()
