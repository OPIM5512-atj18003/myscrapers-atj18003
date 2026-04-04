# Decision Tree: train on all data < today (local TZ); hold out today
# HTTP entrypoint: train_dt_http

import os, io, json, logging, traceback, re
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "preds")            # e.g., "structured/preds"
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")      # split by local day
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

def _upload_file_to_gcs(client: storage.Client, bucket: str, key: str, local_path: str, content_type: str):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_filename(local_path, content_type=content_type)

def _clean_numeric(s: pd.Series) -> pd.Series:
    # Strip $, commas, spaces; keep digits and dot
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def run_once(dry_run: bool = False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics BEFORE counting/dropping ---
    orig_rows = len(df)
    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    valid_repairs = {"none", "minor", "major"}
    df["recent_repairs"] = df["recent_repairs"].where(
        df["recent_repairs"].isin(valid_repairs), other=np.nan
    )

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info("Recent date counts (local): %s", json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}))

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates", "dates": [str(d) for d in unique_dates]}

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] <  today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    train_df = train_df[train_df["price_num"].notna()]
    dropped_for_target = int((df["date_local"] < today_local).sum()) - int(len(train_df))
    logging.info("Train rows after target clean: %d (dropped_for_target=%d)", len(train_df), dropped_for_target)
    logging.info("Holdout rows today (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    # --- Model: make, model, year_num, mileage_num -> price_num ---
    target = "price_num"
    cat_cols = ["make", "model", "seller_urgency", "recent_repairs", "condition", "transmission"]
    num_cols = ["year_num", "mileage_num"]
    feats = cat_cols + num_cols

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    #model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    #pipe = Pipeline([("pre", pre), ("model", model)])

    from tpot import TPOTRegressor
    X_train = train_df[feats]
    y_train = train_df[target]
    X_train_pre = pre.fit_transform(X_train)
    tpot = TPOTRegressor(generations=1, population_size=3, random_state=42)
    
    tpot.fit(X_train_pre, y_train)
    logging.info("Best TPOT pipeline: %s", tpot.fitted_pipeline_)

    # ---- Predict/evaluate on today's holdout (now includes actual price fields) ----
    mae_today = None
    preds_df = pd.DataFrame()
    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_true = holdout_df["price_num"]

        mask = y_true.notna()
        if mask.any():
            X_h = X_h[mask]
            y_true = y_true[mask]

            X_h_pre = pre.transform(X_h)
            y_hat = np.maximum(tpot.predict(X_h_pre), 0)

            cols = ["post_id", "scraped_at", "make", "model", "year", "mileage", "price", "seller_urgency", "recent_repairs", "condition", "transmission"]
            preds_df = holdout_df.loc[mask, cols].copy()
            preds_df["actual_price"] = y_true       # cleaned numeric truth
            preds_df["pred_price"] = np.round(y_hat, 2)

            mae_today = float(mean_absolute_error(y_true, y_hat))

    # ---- Permutation Importance
        from sklearn.inspection import permutation_importance
        
        X_perm = X_h_pre.toarray() if hasattr(X_h_pre, "toarray") else X_h_pre

        result = permutation_importance(
            tpot.fitted_pipeline_,
            X_perm,
            y_true,
            n_repeats=5,
            random_state=42,
            scoring="neg_mean_absolute_error"
        )

        perm_idx_sorted = result.importances_mean.argsort()
        top5_idx = perm_idx_sorted[-5:]
        feature_names = pre.get_feature_names_out()

        logging.info(
            "Top 5 permutation importance features: %s",
            [(feature_names[i], float(result.importances_mean[i])) for i in top5_idx[::-1]]
        )
        
        perm_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values("importance_mean", ascending=False)
        perm_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/permutation_importance.csv"
        _write_csv_to_gcs(client, GCS_BUCKET, perm_key, perm_df)
        logging.info("Wrote permutation importance to gs://%s/%s (%d rows)", GCS_BUCKET, perm_key, len(perm_df))

    # ---- Partial Dependence Plots (PDP) for top 3 features w/ saving to GCS
        from sklearn.inspection import PartialDependenceDisplay
        import matplotlib.pyplot as plt
        
        feature_to_idx = {name: i for i, name in enumerate(feature_names)}
        top_features = [feature_names[i] for i in top5_idx[::-1][:3]]

        for feat in top_features:
            feature_idx = feature_to_idx[feat]
            fig, ax = plt.subplots(figsize=(6, 4))
            PartialDependenceDisplay.from_estimator(
                tpot.fitted_pipeline_,
                X_perm,
                features=[feature_idx],
                feature_names=feature_names,
                ax=ax,
            )
            plt.title(f"Partial Dependence of {feat}")
            plt.tight_layout()
            
            safe_feat = re.sub(r"[^a-zA-Z0-9_]", "_", feat)
            pdp_filename = f"pdp_{safe_feat}.png"
            pdp_local_path = f"/tmp/{pdp_filename}"
            pdp_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/{pdp_filename}"

            plt.savefig(pdp_local_path)
            plt.close(fig)
            _upload_file_to_gcs(client, GCS_BUCKET, pdp_key, pdp_local_path, content_type="image/png")
            logging.info("Wrote PDP for %s to gs://%s/%s", feat, GCS_BUCKET, pdp_key)

    # --- Output path: HOURLY folder structure ---
    out_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/preds.csv"

    if len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, out_key, preds_df)
        logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, out_key, len(preds_df))
    else:
        logging.info("Dry run or no holdout rows; skip write. Would write to gs://%s/%s", GCS_BUCKET, out_key)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "output_key": out_key,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }

def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
