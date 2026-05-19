from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/features/business_features.parquet")
REPORT_PATH = Path("reports/model_comparison.json")
SELECTION_PATH = Path("reports/model_selection.json")
PREDICTIONS_PATH = Path("reports/model_predictions.parquet")

TARGET = "orders_count"


def metrics(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred), 0)
    y_true = np.asarray(y_true)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


def temporal_split(df):
    df = df.sort_values("window_start")
    times = df["window_start"].drop_duplicates().sort_values()

    train_end = times.iloc[int(len(times) * 0.70)]
    val_end = times.iloc[int(len(times) * 0.85)]

    train = df[df["window_start"] <= train_end]
    val = df[(df["window_start"] > train_end) & (df["window_start"] <= val_end)]
    test = df[df["window_start"] > val_end]

    return train, val, test


def feature_columns(df):
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
        raise ValueError("no safe feature columns found")

    return cols


def add_train_history_features(train, val, test):
    global_median = train[TARGET].median()
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
                train.groupby(keys)[TARGET]
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


def train_historical_baseline(train, test):
    baseline = (
        train.groupby(["area", "day_of_week", "hour"])[TARGET]
        .median()
        .reset_index()
        .rename(columns={TARGET: "prediction"})
    )

    result = test.merge(baseline, on=["area", "day_of_week", "hour"], how="left")
    result["prediction"] = result["prediction"].fillna(train[TARGET].median())

    return result["prediction"].values


def train_random_forest(train, test, cols):
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=14,
        min_samples_leaf=10,
        max_samples=0.4,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(train[cols], train[TARGET])
    return model.predict(test[cols])


def train_xgboost(train, val, test, cols):
    model = xgb.XGBRegressor(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[cols],
        train[TARGET],
        eval_set=[(val[cols], val[TARGET])],
        verbose=False,
    )

    return model.predict(test[cols])


def train_lightgbm(train, val, test, cols):
    train_data = lgb.Dataset(train[cols], label=train[TARGET])
    val_data = lgb.Dataset(val[cols], label=val[TARGET])

    candidates = [
        (
            "l1_small",
            {
                "objective": "regression_l1",
                "metric": "mae",
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
            },
            1200,
        ),
        (
            "huber",
            {
                "objective": "huber",
                "metric": "mae",
                "alpha": 0.9,
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
            },
            1200,
        ),
        (
            "poisson",
            {
                "objective": "poisson",
                "metric": "mae",
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
            },
            1200,
        ),
    ]

    best = None
    for candidate_name, params, rounds in candidates:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=rounds,
            callbacks=[
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        val_pred = np.maximum(model.predict(val[cols], num_iteration=model.best_iteration), 0)
        val_mae = mean_absolute_error(val[TARGET], val_pred)
        val_rmse = root_mean_squared_error(val[TARGET], val_pred)
        print(
            f"  LightGBM {candidate_name}: "
            f"val_mae={val_mae:.4f}, val_rmse={val_rmse:.4f}, "
            f"best_iter={model.best_iteration}"
        )

        if best is None or val_mae < best["val_mae"]:
            best = {
                "name": candidate_name,
                "params": params,
                "best_iteration": model.best_iteration,
                "val_mae": val_mae,
            }

    print(f"  Selected LightGBM config: {best['name']}")
    full_train = pd.concat([train, val], ignore_index=True)
    final_model = lgb.train(
        best["params"],
        lgb.Dataset(full_train[cols], label=full_train[TARGET]),
        num_boost_round=best["best_iteration"],
    )

    return final_model.predict(test[cols])


def select_best_ml_model(results):
    ml_results = {
        name: scores
        for name, scores in results.items()
        if name != "historical_baseline"
    }
    if not ml_results:
        raise ValueError("no ML model results found")

    return {
        "best_ml_by_mae": min(ml_results, key=lambda name: ml_results[name]["mae"]),
        "best_ml_by_rmse": min(ml_results, key=lambda name: ml_results[name]["rmse"]),
        "best_ml_by_mape": min(ml_results, key=lambda name: ml_results[name]["mape"]),
    }


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    train, val, test = temporal_split(df)
    train, val, test = add_train_history_features(train, val, test)
    cols = feature_columns(pd.concat([train, val, test], ignore_index=True))

    results = {}
    prediction_rows = []

    models = {
        "historical_baseline": lambda: train_historical_baseline(train, test),
        "random_forest": lambda: train_random_forest(train, test, cols),
        "xgboost": lambda: train_xgboost(train, val, test, cols),
        "lightgbm": lambda: train_lightgbm(train, val, test, cols),
    }

    for name, fn in models.items():
        print(f"Training/evaluating {name}...")
        started_at = time.perf_counter()
        preds = fn()
        results[name] = metrics(test[TARGET].values, preds)
        print(f"{name} finished in {time.perf_counter() - started_at:.1f}s")

        tmp = test[["window_start", "area", TARGET]].copy()
        tmp["model"] = name
        tmp["prediction"] = preds
        prediction_rows.append(tmp)

    comparison = pd.DataFrame(results).T.sort_values("mae")
    selection = select_best_ml_model(results)

    print("\nMODEL COMPARISON")
    print(comparison)
    print("\nBEST ML MODEL")
    print(selection)

    with REPORT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    with SELECTION_PATH.open("w") as f:
        json.dump(selection, f, indent=2)

    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )


if __name__ == "__main__":
    main()
