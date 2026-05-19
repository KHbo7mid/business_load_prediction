from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

FEATURES = Path("data/features/business_features.parquet")
MODEL_DIR = Path("models")
REPORTS = Path("reports")

TARGETS = ["orders_count", "driver_requests_count"]


def temporal_split(df):
    df = df.sort_values("window_start")
    times = df["window_start"].drop_duplicates().sort_values()

    train_end = times.iloc[int(len(times) * 0.70)]
    val_end = times.iloc[int(len(times) * 0.85)]

    train = df[df["window_start"] <= train_end]
    val = df[(df["window_start"] > train_end) & (df["window_start"] <= val_end)]
    test = df[df["window_start"] > val_end]

    return train, val, test


def metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    mask = y_true > 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        positive_mae = mean_absolute_error(y_true[mask], y_pred[mask])
    else:
        mape = np.nan
        positive_mae = np.nan

    zero_mask = y_true == 0
    zero_mae = mean_absolute_error(y_true[zero_mask], y_pred[zero_mask]) if zero_mask.any() else np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "positive_mae": float(positive_mae),
        "zero_mae": float(zero_mae),
    }


def training_weights(y):
    positive = y > 0
    positive_count = int(positive.sum())
    zero_count = int((~positive).sum())
    if positive_count == 0:
        return None

    positive_weight = min(max(zero_count / positive_count, 1.0), 10.0)
    weights = np.ones(len(y), dtype="float32")
    weights[positive.to_numpy()] = positive_weight
    return weights


def selection_score(result_metrics):
    return (result_metrics["mae"] + result_metrics["positive_mae"]) / 2


def feature_columns(df, target):
    calendar_features = {
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    }
    safe_history_patterns = (
        "_lag_",
        "_roll_",
        "_velocity_",
        "_ewm_",
    )

    cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if (
            c in calendar_features
            or c.startswith("history_")
            or any(pattern in c for pattern in safe_history_patterns)
        ):
            cols.append(c)

    if not cols:
        raise ValueError(f"no safe feature columns found for {target}")

    return cols


def add_train_history_features(train, val, test, target):
    global_median = train[target].median()
    group_specs = [
        (["area"], "history_area"),
        (["area", "hour"], "history_area_hour"),
        (["area", "day_of_week", "hour"], "history_area_dow_hour"),
        (["day_of_week", "hour"], "history_dow_hour"),
    ]

    enriched = []
    for frame in (train, val, test):
        out = frame.copy()
        for keys, prefix in group_specs:
            summary = (
                train.groupby(keys)[target]
                .agg(
                    median="median",
                    mean="mean",
                    p75=lambda s: s.quantile(0.75),
                    std="std",
                )
                .reset_index()
            )
            summary = summary.rename(
                columns={
                    "median": f"{prefix}_median",
                    "mean": f"{prefix}_mean",
                    "p75": f"{prefix}_p75",
                    "std": f"{prefix}_std",
                }
            )
            out = out.merge(summary, on=keys, how="left")
            history_cols = [
                f"{prefix}_median",
                f"{prefix}_mean",
                f"{prefix}_p75",
                f"{prefix}_std",
            ]
            out[history_cols] = out[history_cols].fillna(
                {
                    f"{prefix}_median": global_median,
                    f"{prefix}_mean": global_median,
                    f"{prefix}_p75": global_median,
                    f"{prefix}_std": 0,
                }
            )
        enriched.append(out)

    return tuple(enriched)


def lightgbm_candidates():
    base = {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 8,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbose": -1,
    }
    return [
        ("l1", {**base, "objective": "regression_l1", "metric": "mae"}),
        ("huber", {**base, "objective": "huber", "metric": "mae", "alpha": 0.9}),
        ("poisson", {**base, "objective": "poisson", "metric": "mae"}),
    ]


def train_target(df, target):
    train, val, test = temporal_split(df)
    train, val, test = add_train_history_features(train, val, test, target)
    cols = feature_columns(pd.concat([train, val, test], ignore_index=True), target)

    train_data = lgb.Dataset(
        train[cols],
        label=train[target],
        weight=training_weights(train[target]),
        feature_name=cols,
    )
    val_data = lgb.Dataset(val[cols], label=val[target], feature_name=cols)

    best = None
    for config_name, params in lightgbm_candidates():
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1200,
            callbacks=[
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        val_preds = np.maximum(model.predict(val[cols], num_iteration=model.best_iteration), 0)
        val_metrics = metrics(val[target].values, val_preds)
        print(
            f"{target} / {config_name}: "
            f"val_mae={val_metrics['mae']:.4f}, "
            f"val_rmse={val_metrics['rmse']:.4f}, "
            f"best_iter={model.best_iteration}"
        )

        if best is None or selection_score(val_metrics) < selection_score(best["valMetrics"]):
            best = {
                "configName": config_name,
                "params": params,
                "bestIteration": model.best_iteration,
                "valMetrics": val_metrics,
            }

    full_train = pd.concat([train, val], ignore_index=True)
    final_model = lgb.train(
        best["params"],
        lgb.Dataset(
            full_train[cols],
            label=full_train[target],
            weight=training_weights(full_train[target]),
            feature_name=cols,
        ),
        num_boost_round=best["bestIteration"],
    )

    preds = final_model.predict(test[cols])
    result_metrics = metrics(test[target].values, preds)

    model_path = MODEL_DIR / f"lgbm_{target}.txt"
    final_model.save_model(str(model_path))

    metadata = {
        "target": target,
        "modelPath": str(model_path),
        "features": cols,
        "featureCount": len(cols),
        "selectedConfig": best["configName"],
        "params": best["params"],
        "validationMetrics": best["valMetrics"],
        "metrics": result_metrics,
        "bestIteration": best["bestIteration"],
    }

    with (MODEL_DIR / f"lgbm_{target}_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES)
    all_results = {}

    for target in TARGETS:
        all_results[target] = train_target(df, target)

    with (REPORTS / "training_report.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
