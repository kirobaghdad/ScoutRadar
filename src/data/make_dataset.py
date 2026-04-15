import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

def load_data(input_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from the given directory into a dictionary of DataFrames."""
    input_path = Path(input_dir)
    tables = {}
    for csv_file in input_path.glob("*.csv"):
        try:
            logger.info(f"Loading {csv_file.name}...")
            tables[csv_file.stem] = pd.read_csv(csv_file, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to load {csv_file.name}: {e}")
    return tables

def handle_missing_values(df: pd.DataFrame, drop_threshold: float = 0.40) -> pd.DataFrame:
    """Drop columns with too many missing values and impute the rest."""
    # 1. Drop columns with missing > threshold
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct[missing_pct > drop_threshold].index
    if not cols_to_drop.empty:
        logger.info(f"Dropping high-missingness columns (>40%): {list(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    
    # 2. Impute remaining missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")
    return df

def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize datetimes and fix ID columns."""
    # Convert obvious date columns
    date_columns = [col for col in df.columns if "date" in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        # Forward fill dates if coerced to NaT (just a standard safeguard)
        if df[col].isna().sum() < len(df) * 0.5:
             df[col] = df[col].ffill()
    
    return df

def cap_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Cap numerical outliers using the IQR bounds, rather than dropping them."""
    for col in df.columns:
        # We only cap continuous numerical distributions; avoid IDs or binary logic
        if pd.api.types.is_numeric_dtype(df[col]) and "id" not in col.lower() and "is_" not in col.lower():
            # If unique values < 5, it's probably categorical encoded, skip it
            if df[col].nunique() < 5:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
                
            lower_bound = q1 - (multiplier * iqr)
            upper_bound = q3 + (multiplier * iqr)
            
            # Clip bounds
            outlier_count = df[col].lt(lower_bound).sum() + df[col].gt(upper_bound).sum()
            if outlier_count > 0:
                logger.debug(f"Capping {outlier_count} IQR outliers in '{col}'")
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def clean_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run the entire cleaning pipeline on a single DataFrame."""
    logger.info(f"--- Cleaning {name} ---")
    size_orig = len(df)
    
    df = df.drop_duplicates()
    if len(df) < size_orig:
        logger.info(f"Dropped {size_orig - len(df)} duplicate rows")
        
    df = standardize_data_types(df)
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df)
    
    return df

def process_data(input_dir: str, output_dir: str):
    """End-to-End data processing pipeline."""
    tables = load_data(input_dir)
    if not tables:
        logger.warning(f"No tables found inside {input_dir}")
        return
        
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    
    for name, df in tables.items():
        if df.empty:
            logger.warning(f"Table {name} is empty, skipping.")
            continue
            
        clean_df = clean_table(name, df)
        
        # Save output
        out_path = output_p / f"{name}_clean.csv"
        clean_df.to_csv(out_path, index=False)
        logger.info(f"Saved {name} to {out_path}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    root_dir = Path(__file__).resolve().parents[2]
    raw_dir = root_dir / "data" / "player_scores_data"
    processed_dir = root_dir / "data" / "processed"
    
    process_data(str(raw_dir), str(processed_dir))
