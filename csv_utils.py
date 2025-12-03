import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple


def csv_to_df(file_path: str, delimiter: str = ';', encoding: str = 'utf-8') -> pd.DataFrame:
    """Load CSV file with automatic encoding detection."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    for enc in [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=enc)
            print(f"Loaded: {file_path} | Shape: {df.shape} | Encoding: {enc}")
            return df
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
    
    raise ValueError(f"Could not read {file_path} with any supported encoding")


def check_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None) -> dict:
    """Check for duplicate rows in DataFrame."""
    total = len(df)
    dups = df.duplicated(subset=columns) if columns else df.duplicated()
    dup_count = dups.sum()
    dup_pct = round((dup_count / total) * 100 if total > 0 else 0, 2)
    dup_indices = df[dups].index.tolist()
    
    cols_info = f"columns {columns}" if columns else "all columns"
    print(f"Duplicates ({cols_info}): {dup_count}/{total} ({dup_pct}%)")
    
    return {
        'total_rows': total,
        'duplicate_rows': dup_count,
        'unique_rows': total - dup_count,
        'duplicate_percentage': dup_pct,
        'duplicate_indices': dup_indices,
    }


def remove_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
    """Remove duplicate rows from DataFrame."""
    original = len(df)
    df_clean = df.drop_duplicates(subset=columns, keep=keep) if columns else df.drop_duplicates(keep=keep)
    removed = original - len(df_clean)
    print(f"Removed {removed} duplicates: {original} → {len(df_clean)} rows")
    return df_clean


def analyze_csv(file_path: str, text_column: Optional[str] = None, delimiter: str = ';') -> Tuple[pd.DataFrame, dict]:
    """Load CSV and check for duplicates."""
    df = csv_to_df(file_path, delimiter=delimiter)
    cols = [text_column] if text_column and text_column in df.columns else None
    dup_info = check_duplicates(df, cols)
    return df, dup_info


if __name__ == "__main__":
    try:
        df, info = analyze_csv("data-2/1prompts_v2.csv", text_column="Prompt")
        print(f"Found {info['duplicate_rows']} duplicates")
    except Exception as e:
        print(f"Error: {e}")