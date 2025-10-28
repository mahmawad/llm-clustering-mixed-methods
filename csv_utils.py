#!/usr/bin/env python3
"""
Simple CSV utilities for loading and checking duplicates.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple


def csv_to_df(file_path: str, delimiter: str = ';', encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Convert CSV file to pandas DataFrame with error handling.
    
    Args:
        file_path: Path to the CSV file
        delimiter: CSV delimiter (default: ';')
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        pandas DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file can't be read with any encoding
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try different encodings
    encodings = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=enc)
            print(f"   - Successfully loaded CSV: {file_path}")
            print(f"   - Shape: {df.shape}")
            print(f"   - Encoding: {enc}")
            print(f"   - Columns: {list(df.columns)}")
            return df
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
        except Exception as e:
            print(f" *- Error with encoding {enc}: {e}")
            continue
    
    raise ValueError(f"Could not read file {file_path} with any supported encoding")


def check_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None) -> dict:
    """
    Check for duplicate rows in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to check for duplicates. If None, checks all columns.
        
    Returns:
        Dictionary with duplicate information:
        - 'total_rows': Total number of rows
        - 'duplicate_rows': Number of duplicate rows
        - 'unique_rows': Number of unique rows
        - 'duplicate_percentage': Percentage of duplicates
        - 'duplicate_indices': Indices of duplicate rows
    """
    total_rows = len(df)
    
    if columns is None:
        # Check duplicates across all columns
        duplicates = df.duplicated()
        subset_info = "all columns"
    else:
        # Check duplicates in specific columns
        duplicates = df.duplicated(subset=columns)
        subset_info = f"columns: {columns}"
    
    duplicate_count = duplicates.sum()
    unique_count = total_rows - duplicate_count
    duplicate_percentage = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
    
    # Get indices of duplicate rows
    duplicate_indices = df[duplicates].index.tolist()
    
    result = {
        'total_rows': total_rows,
        'duplicate_rows': duplicate_count,
        'unique_rows': unique_count,
        'duplicate_percentage': round(duplicate_percentage, 2),
        'duplicate_indices': duplicate_indices,
        'checked_columns': subset_info
    }
    
    # Print summary
    print(f"\n - Duplicate Check Results ({subset_info}):")
    print(f"   - Total rows: {total_rows}")
    print(f"   - Duplicate rows: {duplicate_count}")
    print(f"   - Unique rows: {unique_count}")
    print(f"   - Duplicate percentage: {duplicate_percentage:.2f}%")

    if duplicate_count > 0:
        print(f"    - Duplicate indices: {duplicate_indices[:10]}{'...' if len(duplicate_indices) > 10 else ''}")
    else:
        print("     - No duplicates found!")
    
    return result


def remove_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to check for duplicates. If None, checks all columns.
        keep: Which duplicates to keep ('first', 'last', False)
        
    Returns:
        DataFrame with duplicates removed
    """
    original_count = len(df)
    
    if columns is None:
        df_clean = df.drop_duplicates(keep=keep)
        subset_info = "all columns"
    else:
        df_clean = df.drop_duplicates(subset=columns, keep=keep)
        subset_info = f"columns: {columns}"
    
    removed_count = original_count - len(df_clean)
    
    print(f"\n - Duplicate Removal ({subset_info}):")
    print(f"   - Original rows: {original_count}")
    print(f"   - Removed rows: {removed_count}")
    print(f"   - Final rows: {len(df_clean)}")

    return df_clean


def analyze_csv(file_path: str, text_column: Optional[str] = None, delimiter: str = ';') -> Tuple[pd.DataFrame, dict]:
    """
    Complete analysis: load CSV and check for duplicates.
    
    Args:
        file_path: Path to CSV file
        text_column: Specific text column to check for duplicates
        delimiter: CSV delimiter
        
    Returns:
        Tuple of (DataFrame, duplicate_info)
    """
    print("=" * 50)
    print("📁 CSV ANALYSIS")
    print("=" * 50)
    
    # Load CSV
    df = csv_to_df(file_path, delimiter=delimiter)
    
    # Check duplicates
    if text_column and text_column in df.columns:
        duplicate_info = check_duplicates(df, columns=[text_column])
    else:
        duplicate_info = check_duplicates(df)
    
    print("\n" + "=" * 50)
    
    return df, duplicate_info


# Example usage functions
def example_usage():
    """Example of how to use the CSV utilities."""
    
    # Example 1: Simple CSV loading
    df = csv_to_df("data-2/1prompts_v2.csv")
    
    # Example 2: Check for duplicates in all columns
    duplicate_info = check_duplicates(df)
    
    # Example 3: Check for duplicates in specific column
    if 'Prompt' in df.columns:
        text_duplicates = check_duplicates(df, columns=['Prompt'])
    
    # Example 4: Remove duplicates
    df_clean = remove_duplicates(df, columns=['Prompt'])
    
    # Example 5: Complete analysis
    df, info = analyze_csv("data-2/1prompts_v2.csv", text_column="Prompt")
    
    return df, info


if __name__ == "__main__":
    # Run example
    try:
        df, duplicate_info = example_usage()
        print(f"\n - Analysis complete! Found {duplicate_info['duplicate_rows']} duplicates.")
    except Exception as e:
        print(f"- Error: {e}")