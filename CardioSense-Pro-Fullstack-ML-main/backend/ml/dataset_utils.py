from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ml.constants import FEATURE_ORDER, INTEGER_FEATURES

TARGET_ALIASES = [
    "target",
    "num",
    "heartdisease",
    "diagnosis",
    "label",
    "class",
    "output",
    "condition",
]

COLUMN_ALIASES = {
    "age": ["age"],
    "sex": ["sex", "gender"],
    "cp": ["cp", "chestpain", "chestpaintype", "chestpaincategory"],
    "trestbps": [
        "trestbps",
        "restingbp",
        "restingbloodpressure",
        "restbp",
    ],
    "chol": ["chol", "cholesterol", "serumcholesterol"],
    "fbs": ["fbs", "fastingbs", "fastingbloodsugar"],
    "restecg": ["restecg", "restingecg", "restingelectrocardiographicresults"],
    "thalach": ["thalach", "maxhr", "maxheartrate", "maximumheartrate"],
    "exang": ["exang", "exerciseangina", "exerciseinducedangina"],
    "oldpeak": ["oldpeak", "stdepression"],
    "slope": ["slope", "stslope", "stsegment", "slopeofpeakexercise"],
    "ca": ["ca", "majorvessels", "nummajorvessels", "vessels"],
    "thal": ["thal", "thalassemia", "thalium"],
}


def _normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        normalized = _normalize_column_name(alias)
        if normalized in df.columns:
            return normalized
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_binary(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    if numeric.notna().mean() > 0.7:
        return (numeric > 0).astype(float)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "n": 0,
        "y": 1,
        "negative": 0,
        "positive": 1,
        "healthy": 0,
        "disease": 1,
        "absent": 0,
        "present": 1,
    }
    return text.map(mapping).astype(float)


def _map_sex(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    if numeric.notna().mean() > 0.7:
        return numeric.clip(lower=0, upper=1)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "m": 1,
        "male": 1,
        "f": 0,
        "female": 0,
        "1": 1,
        "0": 0,
    }
    return text.map(mapping).astype(float)


def _map_cp(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    if numeric.notna().mean() > 0.7:
        if numeric.dropna().between(1, 4).all():
            numeric = numeric - 1
        return numeric.clip(lower=0, upper=3)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "ta": 0,
        "typicalangina": 0,
        "typical": 0,
        "ata": 1,
        "atypicalangina": 1,
        "atypical": 1,
        "nap": 2,
        "nonanginal": 2,
        "nonanginalpain": 2,
        "asy": 3,
        "asymptomatic": 3,
    }
    return text.map(mapping).astype(float)


def _map_restecg(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    if numeric.notna().mean() > 0.7:
        return numeric.clip(lower=0, upper=2)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "normal": 0,
        "st": 1,
        "stt": 1,
        "sttwaveabnormality": 1,
        "lvh": 2,
        "leftventricularhypertrophy": 2,
    }
    return text.map(mapping).astype(float)


def _map_slope(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    if numeric.notna().mean() > 0.7:
        if numeric.dropna().between(1, 3).all():
            numeric = numeric - 1
        return numeric.clip(lower=0, upper=2)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "up": 0,
        "upsloping": 0,
        "flat": 1,
        "down": 2,
        "downsloping": 2,
    }
    return text.map(mapping).astype(float)


def _map_exang(series: pd.Series) -> pd.Series:
    return _to_binary(series)


def _map_thal(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    if numeric.notna().mean() > 0.7:
        rounded = numeric.round()
        if rounded.dropna().isin([3, 6, 7]).all():
            mapping = {3: 0, 6: 1, 7: 2}
            return rounded.map(mapping).fillna(3).astype(float)
        return rounded.clip(lower=0, upper=3)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "normal": 0,
        "fixed": 1,
        "fixeddefect": 1,
        "reversible": 2,
        "reversibledefect": 2,
        "unknown": 3,
    }
    return text.map(mapping).fillna(3).astype(float)


def _map_target(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    if numeric.notna().mean() > 0.7:
        return (numeric > 0).astype(float)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "negative": 0,
        "positive": 1,
        "healthy": 0,
        "disease": 1,
        "absence": 0,
        "presence": 1,
    }
    return text.map(mapping).astype(float)


def canonicalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: _normalize_column_name(column)
        for column in raw_df.columns
    }
    df = raw_df.rename(columns=renamed).copy()

    target_col = _find_column(df, TARGET_ALIASES)
    if target_col is None:
        raise ValueError("Target column was not found in dataset")

    canonical: dict[str, pd.Series] = {}

    for feature in FEATURE_ORDER:
        source_col = _find_column(df, COLUMN_ALIASES[feature])
        if source_col is None:
            canonical[feature] = pd.Series([np.nan] * len(df))
            continue

        source_series = df[source_col]
        if feature == "sex":
            canonical[feature] = _map_sex(source_series)
        elif feature == "cp":
            canonical[feature] = _map_cp(source_series)
        elif feature == "fbs":
            canonical[feature] = _to_binary(source_series)
        elif feature == "restecg":
            canonical[feature] = _map_restecg(source_series)
        elif feature == "exang":
            canonical[feature] = _map_exang(source_series)
        elif feature == "slope":
            canonical[feature] = _map_slope(source_series)
        elif feature == "thal":
            canonical[feature] = _map_thal(source_series)
        else:
            canonical[feature] = _to_numeric(source_series)

    target_series = _map_target(df[target_col])

    canonical_df = pd.DataFrame(canonical)
    canonical_df["target"] = target_series
    canonical_df = canonical_df.dropna(subset=["target"]).copy()

    if canonical_df.empty:
        raise ValueError("No valid rows remained after canonicalization")

    fill_defaults = {
        "ca": 0,
        "thal": 3,
    }

    for feature in FEATURE_ORDER:
        if feature in fill_defaults:
            canonical_df[feature] = canonical_df[feature].fillna(fill_defaults[feature])
            continue

        if canonical_df[feature].notna().any():
            median_value = canonical_df[feature].median()
            canonical_df[feature] = canonical_df[feature].fillna(median_value)
        else:
            canonical_df[feature] = canonical_df[feature].fillna(0)

    for feature in INTEGER_FEATURES:
        canonical_df[feature] = canonical_df[feature].round().astype(int)

    canonical_df["oldpeak"] = canonical_df["oldpeak"].astype(float)
    canonical_df["target"] = canonical_df["target"].round().astype(int)

    canonical_df = canonical_df.replace([np.inf, -np.inf], np.nan).dropna()
    return canonical_df.reset_index(drop=True)


def load_training_frame(raw_data_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    csv_files = sorted(raw_data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    frames: list[pd.DataFrame] = []
    sources: list[Path] = []

    for csv_file in csv_files:
        try:
            raw_df = pd.read_csv(csv_file)
            normalized_df = canonicalize_dataframe(raw_df)
            if len(normalized_df) < 50:
                continue
            frames.append(normalized_df)
            sources.append(csv_file)
        except Exception:
            continue

    if not frames:
        raise ValueError("No usable training data was found after normalization")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)

    if merged["target"].nunique() < 2:
        raise ValueError("Training data must contain both classes")

    return merged, sources
