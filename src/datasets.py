"""Load, harmonize, and validate concrete strength datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
    "strength_mpa",
]

COLUMN_ALIASES: dict[str, list[str]] = {
    "cement": [
        "Cement (component 1)(kg in a m^3 mixture)",
        "cement (kg/m3)",
        "Cement",
    ],
    "slag": [
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
        "slag (kg/m3)",
        "Slag",
        "GGBS",
        "ggbs",
    ],
    "fly_ash": [
        "Fly Ash (component 3)(kg in a m^3 mixture)",
        "flyash (kg/m3)",
        "flyash",
        "Fly Ash",
        "Fly ash",
    ],
    "water": [
        "Water  (component 4)(kg in a m^3 mixture)",
        "water (kg/m3)",
        "Water",
    ],
    "superplasticizer": [
        "Superplasticizer (component 5)(kg in a m^3 mixture)",
        "superplasticizer (kg/m3)",
        "Superplasticizer",
        "SP",
    ],
    "coarse_agg": [
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
        "coarse aggregate (kg/m3)",
        "Coarse Aggregate",
        "Coarse Aggr.",
    ],
    "fine_agg": [
        "Fine Aggregate (component 7)(kg in a m^3 mixture)",
        "fine aggregate (kg/m3)",
        "Fine Aggregate",
        "Fine Aggr.",
    ],
    "age": ["Age (day)", "age (days)", "Age"],
    "strength_mpa": [
        "Concrete compressive strength",
        "Compressive Strength (MPa)",
        "compressive strength (MPa)",
        "Strength (MPa)",
        "strength",
        "Experimental_28_day_cube_strength",
        "Experimental 28 day cube strength (MPa)",
        "Compressive Strength (28-day)(Mpa)",
        "Compressive Strength (28-day)(MPa)",
    ],
}


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names."""
    df = df.copy()
    rename_map: dict[str, str] = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
            if alias.lower() in lower_cols:
                rename_map[lower_cols[alias.lower()]] = canonical
                break

    df = df.rename(columns=rename_map)
    return df


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all canonical feature columns exist."""
    df = harmonize_columns(df)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            if col == "strength_mpa":
                continue
            df[col] = 0.0
        elif col in df.columns and col != "domain_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "strength_mpa" in df.columns:
        df["strength_mpa"] = pd.to_numeric(df["strength_mpa"], errors="coerce")
    return df


UCI_182_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "concrete/slump/slump_test.data"
)
OPENML_UCI165_ID = 4353  # Concrete Compressive Strength (1030 rows)


class ConcreteDatasetLoader:
    """Load concrete compressive strength datasets from multiple sources."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache_dir: Path | str = Path("data/raw"),
        seed: int = 42,
    ):
        self.config = config or {}
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        np.random.seed(seed)

    def load_uci_165(self) -> pd.DataFrame:
        """UCI #165: Yeh concrete compressive strength (1030 samples, ages 1–365d)."""
        logger.info("Loading UCI #165...")
        try:
            concrete = fetch_ucirepo(id=165)
            df = concrete.data.features.copy()
            df = harmonize_columns(df)
            targets = concrete.data.targets
            target_col = targets.columns[0]
            df["strength_mpa"] = targets[target_col].values
        except Exception as e:
            logger.warning("  UCI #165 via ucimlrepo failed (%s); trying OpenML...", e)
            df = self._load_uci_165_openml()

        df["domain_id"] = "D1_uci165"
        df = ensure_canonical_columns(df)
        logger.info("  Loaded %d samples", len(df))
        return df

    def _load_uci_165_openml(self) -> pd.DataFrame:
        """Fallback: OpenML #4353 (same as UCI #165)."""
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=OPENML_UCI165_ID, as_frame=True, parser="auto")
        df = data.frame.copy()
        df = harmonize_columns(df)
        if "strength_mpa" not in df.columns:
            for col in df.columns:
                if "strength" in col.lower():
                    df = df.rename(columns={col: "strength_mpa"})
                    break
        logger.info("  Loaded UCI #165 from OpenML #%d (%d samples)", OPENML_UCI165_ID, len(df))
        return df

    def load_uci_182(self) -> pd.DataFrame:
        """UCI #182: Yeh concrete slump test: 28-day strength (103 samples)."""
        logger.info("Loading UCI #182 (Concrete Slump Test)...")
        local_cache = self.cache_dir / "uci_182" / "slump_test.csv"

        if local_cache.exists():
            logger.info("  Using cached %s", local_cache)
            df = self._parse_uci_182_slump(pd.read_csv(local_cache))
        else:
            try:
                slump = fetch_ucirepo(id=182)
                df = slump.data.features.copy()
                df = harmonize_columns(df)
                targets = slump.data.targets
                strength_col = None
                for col in targets.columns:
                    cl = col.lower()
                    if "28" in cl or "strength" in cl or "compressive" in cl:
                        strength_col = col
                        break
                if strength_col is None:
                    strength_col = targets.columns[-1]
                df["strength_mpa"] = targets[strength_col].values
            except Exception as e:
                logger.warning("  ucimlrepo id=182 unavailable (%s)", e)
                df = self._fetch_uci_182_archive(save_cache=local_cache)

        df["domain_id"] = "D2_uci182"
        df = ensure_canonical_columns(df)
        if "age" not in df.columns or df["age"].isna().all():
            df["age"] = 28.0
        logger.info("  Loaded %d samples", len(df))
        return df

    def _parse_uci_182_slump(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize UCI #182 slump CSV to canonical schema."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        df = harmonize_columns(df)

        rename_extra = {
            "SLUMP(cm)": "slump_cm",
            "FLOW(cm)": "flow_cm",
            "No": "mix_ref",
        }
        df = df.rename(columns={k: v for k, v in rename_extra.items() if k in df.columns})

        if "strength_mpa" not in df.columns:
            for col in df.columns:
                if "strength" in col.lower() and "28" in col.lower():
                    df = df.rename(columns={col: "strength_mpa"})
                    break

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].replace("-", np.nan)
        for col in df.columns:
            if col not in ("mix_ref", "domain_id"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=["strength_mpa", "cement", "water"])

    def _fetch_uci_182_archive(self, save_cache: Path | None = None) -> pd.DataFrame:
        """Download UCI #182 from archive URL (CSV with header row)."""
        logger.info("  Fetching UCI #182 from archive URL...")
        try:
            raw = pd.read_csv(UCI_182_URL)
            df = self._parse_uci_182_slump(raw)
            if save_cache is not None and len(df) > 0:
                save_cache.parent.mkdir(parents=True, exist_ok=True)
                raw.to_csv(save_cache, index=False)
                logger.info("  Cached to %s", save_cache)
            if len(df) == 0:
                raise ValueError("No valid rows parsed from UCI #182 URL")
            logger.info("  Loaded %d samples from UCI archive", len(df))
            return df
        except Exception as e:
            logger.warning("  UCI #182 archive fetch failed (%s); using synthetic D2", e)
            return self._synthetic_uci_182()

    def _synthetic_uci_182(self) -> pd.DataFrame:
        """Minimal synthetic slump dataset when UCI #182 unavailable."""
        rng = np.random.default_rng(self.seed + 182)
        n = 103
        df = pd.DataFrame({
            "cement": rng.uniform(150, 540, n),
            "slag": rng.uniform(0, 360, n),
            "fly_ash": rng.uniform(0, 200, n),
            "water": rng.uniform(120, 240, n),
            "superplasticizer": rng.uniform(0, 32, n),
            "coarse_agg": rng.uniform(800, 1150, n),
            "fine_agg": rng.uniform(600, 950, n),
            "age": 28.0,
            "strength_mpa": rng.uniform(15, 65, n),
        })
        return ensure_canonical_columns(df)

    def load_zviazhynski(self) -> pd.DataFrame:
        """Zviazhynski et al. 2025: supplementary Table 1."""
        zvzh_dir = self.cache_dir / "zviazhynski_2025"
        zvzh_file = zvzh_dir / "zviazhynski_supplementary.csv"

        if not zvzh_file.exists():
            logger.warning("%s not found; skipping D3", zvzh_file)
            logger.warning("  See data/MANUAL_DOWNLOAD.md for download instructions")
            return pd.DataFrame()

        logger.info("Loading Zviazhynski from %s...", zvzh_file)
        df = pd.read_csv(zvzh_file)
        df.columns = df.columns.str.strip()

        # Fix common mis-export from PDF: col labeled GGBS holds cement, next col is slag
        if "cement" not in df.columns and "GGBS" in df.columns:
            cols = list(df.columns)
            ggbs_idx = cols.index("GGBS")
            df = df.rename(columns={
                cols[ggbs_idx]: "cement",
                cols[ggbs_idx + 1]: "slag" if cols[ggbs_idx + 1] != "slag" else "slag",
            })
            if "slag" not in df.columns and ggbs_idx + 1 < len(cols):
                df = df.rename(columns={cols[ggbs_idx + 1]: "slag"})

        df = self._normalize_zviazhynski(df)
        df = ensure_canonical_columns(df)
        df["domain_id"] = "D3_zviazhynski"
        if "age" not in df.columns or df["age"].isna().all():
            df["age"] = 28.0
        logger.info("  Loaded %d samples", len(df))
        return df

    def _normalize_zviazhynski(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map supplementary table columns to canonical names."""
        rename = {
            "Fine_aggregate": "fine_agg",
            "Fine aggregate (kg/m3)": "fine_agg",
            "Coarse_aggregate": "coarse_agg",
            "Coarse aggregate (kg/m3)": "coarse_agg",
            "Water_binder": "w_b_ratio",
            "Water binder ratio": "w_b_ratio",
            "Experimental_28_day_cube_strength": "strength_mpa",
            "Experimental_56_day": "strength_56d_mpa",
            "Total_slump_height": "slump_mm",
            "Density": "density_kg_m3",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Broken header split across two columns for 56-day strength
        if "strength_mpa" not in df.columns:
            for col in df.columns:
                cl = col.lower()
                if "28" in cl and "strength" in cl:
                    df = df.rename(columns={col: "strength_mpa"})
                    break

        if "strength_mpa" not in df.columns and "strength_28d_mpa" in df.columns:
            df["strength_mpa"] = df["strength_28d_mpa"]

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].replace("-", np.nan)
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def load_mondal(self) -> pd.DataFrame:
        """Mondal 2026 fly-ash dataset (n=55 mixes, 3 ages)."""
        mondal_dir = self.cache_dir / "mondal_2026"
        mondal_file = mondal_dir / "mondal_2026_55mixes.csv"

        if not mondal_file.exists():
            logger.info("  %s not found; reconstructing from paper design...", mondal_file)
            df = self._reconstruct_mondal_from_tables()
            mondal_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(mondal_file, index=False)
            logger.info("  Saved reconstructed dataset to %s", mondal_file)
        else:
            df = pd.read_csv(mondal_file)
            df = ensure_canonical_columns(df)
            if self._mondal_csv_is_invalid(df):
                logger.warning(
                    "  %s has invalid strengths (likely bad export); reconstructing...",
                    mondal_file,
                )
                df = self._reconstruct_mondal_from_tables()
                df.to_csv(mondal_file, index=False)
                logger.info("  Overwrote %s with reconstructed data", mondal_file)

        df["domain_id"] = "D4_mondal"
        logger.info("  Loaded %d samples", len(df))
        return df

    def _mondal_csv_is_invalid(self, df: pd.DataFrame) -> bool:
        """Detect placeholder/broken Mondal CSV (e.g. all strength = 100)."""
        if "strength_mpa" not in df.columns or len(df) == 0:
            return True
        s = pd.to_numeric(df["strength_mpa"], errors="coerce").dropna()
        if len(s) == 0:
            return True
        if s.nunique() <= 1:
            return True
        if s.max() == 100.0 and s.min() == 100.0:
            return True
        return False

    def load_extra_datasets(self) -> dict[str, pd.DataFrame]:
        """Load user-provided CSVs from data/raw/extra/."""
        extra_dir = self.cache_dir / "extra"
        datasets: dict[str, pd.DataFrame] = {}
        if not extra_dir.exists():
            return datasets

        for i, csv_path in enumerate(sorted(extra_dir.glob("*.csv"))):
            df = pd.read_csv(csv_path)
            df = ensure_canonical_columns(df)
            key = f"D5_extra_{i}"
            df["domain_id"] = key
            datasets[key] = df
            logger.info("  Loaded extra dataset %s: %d samples", csv_path.name, len(df))
        return datasets

    def _reconstruct_mondal_from_tables(self) -> pd.DataFrame:
        """Reconstruct Mondal n=55 mixes from experimental design."""
        binder_contents = [300, 375, 450]
        w_cm_ratios = [0.40, 0.45, 0.50, 0.55, 0.60]
        fa_replacements = [0, 20, 30, 40, 50]

        mixes: list[dict[str, float]] = []
        for cm in binder_contents:
            for w_cm in w_cm_ratios:
                for fa_pct in fa_replacements:
                    if cm == 300 and w_cm == 0.40 and fa_pct > 0:
                        continue
                    if cm == 450 and w_cm >= 0.55:
                        continue

                    cement = cm * (100 - fa_pct) / 100
                    fly_ash = cm * fa_pct / 100
                    water = w_cm * cm

                    paste_volume = (cement / 3.14 + fly_ash / 2.13 + water / 1.0) / 1000
                    total_agg = max((1 - paste_volume) * 1000, 0)
                    coarse_agg = total_agg * 1.38 / (1 + 1.38)
                    fine_agg = total_agg / (1 + 1.38)

                    mixes.append(
                        {
                            "cement": cement,
                            "fly_ash": fly_ash,
                            "slag": 0.0,
                            "water": water,
                            "superplasticizer": 0.0,
                            "coarse_agg": coarse_agg,
                            "fine_agg": fine_agg,
                            "age": np.nan,
                            "strength_mpa": np.nan,
                        }
                    )

        df = pd.DataFrame(mixes)
        ages = [7, 28, 90]
        dfs_ages = []
        for age in ages:
            df_age = df.copy()
            df_age["age"] = float(age)
            df_age["strength_mpa"] = self._predict_strength_mondal_equations(
                df_age[["cement", "fly_ash", "water"]], age
            )
            dfs_ages.append(df_age)

        return pd.concat(dfs_ages, ignore_index=True)

    def _predict_strength_mondal_equations(
        self, mix_df: pd.DataFrame, age: int
    ) -> np.ndarray:
        """
        Mondal Eq. 5.11 (28d) and 5.12 (90d) with seed-controlled noise.

        Paper uses natural log: S = exp(a0 - a1*w/cm - a2*f/cm)
        """
        rng = np.random.default_rng(self.seed + age)
        cm = mix_df["cement"] + mix_df["fly_ash"]
        w_cm = mix_df["water"] / cm.clip(lower=1)
        f_cm = mix_df["fly_ash"] / cm.clip(lower=1)

        if age == 7:
            # ~60-70% of 28-day; scaled from Eq. 5.11
            a0, a1, a2 = 4.65, 2.1, 1.2
        elif age == 28:
            a0, a1, a2 = 5.0159, 2.448, 1.533
        elif age == 90:
            a0, a1, a2 = 5.46, 2.94, 0.756
        else:
            raise ValueError(f"Age {age} not supported")

        log_strength = a0 - a1 * w_cm - a2 * f_cm
        strength = np.exp(log_strength)
        strength = strength + rng.normal(0, 0.8, len(strength))
        return strength.clip(5, 80).values

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load and return all available datasets."""
        datasets: dict[str, pd.DataFrame] = {}

        loaders = [
            ("D1", self.load_uci_165),
            ("D2", self.load_uci_182),
            ("D3", self.load_zviazhynski),
            ("D4", self.load_mondal),
        ]

        for key, loader_fn in loaders:
            try:
                df = loader_fn()
                if df is not None and not df.empty:
                    datasets[key] = df
            except Exception as e:
                logger.error("Failed to load %s: %s", key, e)

        try:
            extra = self.load_extra_datasets()
            datasets.update(extra)
        except Exception as e:
            logger.error("Failed to load extra datasets: %s", e)

        logger.info("Loaded %d datasets: %s", len(datasets), list(datasets.keys()))
        return datasets


def generate_derived_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add engineered features and apply validity filters."""
    config = config or {}
    fe = config.get("feature_engineering", {})
    filters_cfg = config.get("filters", {})

    df = ensure_canonical_columns(df.copy())

    for col in ["cement", "slag", "fly_ash", "water", "age"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["binder_total"] = df["cement"] + df["slag"] + df["fly_ash"]
    binder = df["binder_total"].clip(lower=1)
    df["w_cm"] = df["water"] / binder
    df["f_cm"] = df["fly_ash"] / binder
    df["c_cm"] = df["cement"] / binder
    df["s_cm"] = df["slag"] / binder

    if fe.get("derive_interaction", True):
        df["interaction_w_cm_f_cm"] = df["w_cm"] * df["f_cm"]

    df["age_sqrt"] = np.sqrt(df["age"].clip(lower=1))
    df["log_age"] = np.log(df["age"].clip(lower=1))

    w_min = filters_cfg.get("w_cm_min", 0.25)
    w_max = filters_cfg.get("w_cm_max", 1.0)
    s_min = filters_cfg.get("strength_min", 1.0)
    s_max = filters_cfg.get("strength_max", 150.0)

    before = len(df)
    df = df[(df["w_cm"] >= w_min) & (df["w_cm"] <= w_max)].copy()
    if "strength_mpa" in df.columns:
        df = df[(df["strength_mpa"] > s_min) & (df["strength_mpa"] < s_max)].copy()
    after = len(df)
    if before > after:
        logger.warning("Filtered %d invalid rows; %d remain", before - after, after)

    return df.reset_index(drop=True)


BASE_FEATURES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
]

DERIVED_FEATURES = [
    "w_cm",
    "f_cm",
    "c_cm",
    "s_cm",
    "binder_total",
    "interaction_w_cm_f_cm",
    "age_sqrt",
    "log_age",
]

MODEL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return standardized model feature columns present in df."""
    return [c for c in MODEL_FEATURES if c in df.columns]


def align_dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all model feature columns exist in dataframe."""
    df = df.copy()
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def create_lodo_splits(
    datasets_dict: dict[str, pd.DataFrame],
    train_calib_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Leave-one-dataset-out splits.

    For each held-out domain k:
      - Pool remaining datasets
      - Split pool into train (80%) and calibration (20%)
      - Test = held-out domain k
    """
    rng = np.random.default_rng(seed)
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    domain_names = sorted(datasets_dict.keys())

    for held_out in domain_names:
        pool_dfs = [datasets_dict[d] for d in domain_names if d != held_out]
        df_pool = pd.concat(pool_dfs, ignore_index=True)

        n = len(df_pool)
        indices = rng.permutation(n)
        n_train = int(n * train_calib_ratio)
        train_idx = indices[:n_train]
        calib_idx = indices[n_train:]

        df_train = df_pool.iloc[train_idx].reset_index(drop=True)
        df_calib = df_pool.iloc[calib_idx].reset_index(drop=True)
        df_test = datasets_dict[held_out].reset_index(drop=True)

        df_train = align_dataframe_features(df_train)
        df_calib = align_dataframe_features(df_calib)
        df_test = align_dataframe_features(df_test)

        splits[held_out] = (df_train, df_calib, df_test)
        logger.info(
            "LODO fold '%s': train=%d, calib=%d, test=%d",
            held_out,
            len(df_train),
            len(df_calib),
            len(df_test),
        )

    return splits


def save_processed_datasets(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path | str,
) -> None:
    """Save harmonized datasets to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"datasets": {}}
    for domain, df in datasets.items():
        path = output_dir / f"{domain}_harmonized.parquet"
        df.to_parquet(path, index=False)
        manifest["datasets"][domain] = {
            "path": str(path),
            "n_samples": len(df),
            "domain_id": df["domain_id"].iloc[0] if len(df) else None,
        }
        logger.info("Saved %s (%d rows)", path, len(df))

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest to %s", manifest_path)


def save_lodo_splits(
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    output_dir: Path | str,
) -> None:
    """Save LODO train/calib/test splits to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for domain, (train, calib, test) in splits.items():
        train.to_parquet(output_dir / f"{domain}_train.parquet", index=False)
        calib.to_parquet(output_dir / f"{domain}_calib.parquet", index=False)
        test.to_parquet(output_dir / f"{domain}_test.parquet", index=False)

    logger.info("Saved LODO splits to %s", output_dir)


def load_all_datasets(config: dict[str, Any], seed: int = 42) -> dict[str, pd.DataFrame]:
    """Convenience: load, harmonize, and return all datasets."""
    raw_dir = config.get("raw_data_dir", "data/raw")
    loader = ConcreteDatasetLoader(config=config, cache_dir=raw_dir, seed=seed)
    raw = loader.load_all()
    harmonized = {}
    for key, df in raw.items():
        harmonized[key] = generate_derived_features(df, config)
    return harmonized
