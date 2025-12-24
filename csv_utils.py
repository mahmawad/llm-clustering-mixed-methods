import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List, Tuple
from pandas.errors import ParserError


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
            # if parsing fails for this encoding, keep trying the next one
            continue

    raise ValueError(f"Could not read {file_path} with any supported encoding")


def check_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None) -> dict:
    """Check for duplicate rows in DataFrame."""
    total = len(df)
    # decide whether to consider specific columns or the entire row when tracking duplicates
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


def load_csv_with_fallback(csv_file: Path, delimiter: str = ",") -> pd.DataFrame:
    """Try the requested delimiter first, then fall back to other common separators."""
    fallback_delimiters = [",", ";", "\t", "|"]
    candidate_delimiters = []
    if delimiter:
        candidate_delimiters.append(delimiter)
    # ensure we also try other common separators if the preferred one fails
    for alt in fallback_delimiters:
        if alt not in candidate_delimiters:
            candidate_delimiters.append(alt)
    # candidate list now has the desired delimiter followed by reliable fallbacks

    last_error: Optional[ParserError] = None
    for delim in candidate_delimiters:
        try:
            df = pd.read_csv(str(csv_file), delimiter=delim)
            if delim != delimiter:
                print(f"Used fallback delimiter '{delim}' for {csv_file.name}")
            return df
        except ParserError as exc:
            last_error = exc

    message = (
        f"Unable to parse {csv_file.name} with delimiters "
        f"{', '.join(candidate_delimiters)}."
    )
    raise ParserError(message) from last_error


def load_and_prepare_csv(csv_file: Path, text_column: str, delimiter: str = ",") -> pd.DataFrame:
    """Load a CSV, report duplicates, and drop duplicates when possible."""
    # centralize CSV loading logic so main just works with clean data
    df = load_csv_with_fallback(csv_file, delimiter)
    columns = [text_column] if text_column and text_column in df.columns else None
    dup_info = check_duplicates(df, columns)
    if columns and dup_info.get("duplicate_rows", 0) > 0:
        # Keep only the first row for each unique prompt to avoid redundant classifications
        df = df.drop_duplicates(subset=columns)
        # report the new row count after deduplication for the user
        print(f"After removing duplicates: {len(df)} rows")
    return df


def format_display_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def discover_csv_files(script_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for entry in script_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("data"):
            candidates.extend(sorted(entry.glob("*.csv")))
    return sorted(candidates)


def _find_candidate_index(token: str, candidates: List[Path], script_dir: Path) -> Optional[int]:
    for idx, candidate in enumerate(candidates):
        # allow referencing files by filename, absolute path, or user-facing relative path
        if token == candidate.name:
            return idx
        if token == str(candidate):
            return idx
        if token == format_display_path(candidate, script_dir):
            return idx
    return None


def prompt_user_for_files(candidates: List[Path], script_dir: Path) -> List[Path]:
    print("Available data files:")
    for idx, candidate in enumerate(candidates, start=1):
        print(f"  {idx:>2}. {format_display_path(candidate, script_dir)}")

    try:
        selection = input("\nEnter numbers (e.g. 1 3) or 'all' to process every file: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo selection made; defaulting to the first file.")
        return [candidates[0]]

    if not selection:
        return [candidates[0]]
    if selection.lower() in {"all", "*"}:
        return candidates

    # allow comma-separated or space-delimited choices
    tokens = selection.replace(",", " ").split()
    selected: List[Path] = []
    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(candidates):
                selected.append(candidates[idx])
        else:
            found = _find_candidate_index(token, candidates, script_dir)
            if found is not None:
                selected.append(candidates[found])

    if not selected:
        print("No valid selection; defaulting to the first file.")
        return [candidates[0]]

    unique_selected: List[Path] = []
    seen = set()
    for candidate in selected:
        if candidate not in seen:
            unique_selected.append(candidate)
            seen.add(candidate)
    return unique_selected


def resolve_data_file(script_dir: Path, path_str: str) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else script_dir / candidate


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Classify each document in a CSV, allowing interactive file selection."
    )
    parser.add_argument(
        "-f",
        "--data-files",
        nargs="+",
        metavar="FILE",
        help="Path(s) to CSV files (relative to this script or absolute).",
    )
    parser.add_argument(
        "--text-column",
        default="content",
        help="Column to use for classification.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Delimiter used when reading the CSV file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the processing to the first N samples.",
    )
    return parser


if __name__ == "__main__":
    try:
        df, info = analyze_csv("data-2/1prompts_v2.csv", text_column="Prompt")
        print(f"Found {info['duplicate_rows']} duplicates")
    except Exception as e:
        print(f"Error: {e}")
