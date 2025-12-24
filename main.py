from pathlib import Path
from typing import Union

from csv_utils import (
    discover_csv_files,
    format_display_path,
    load_and_prepare_csv,
    parse_args,
    prompt_user_for_files,
    resolve_data_file,
)
from llm_helper import classify_prompt, detect_language, get_selected_category_codes


def main(
    data_file: Union[str, Path],
    text_column: str = "content",
    delimiter: str = ",",
    max_samples: int = None,
):
    print("=" * 60)
    print("LOADING AND ANALYZING DATA")
    print("=" * 60)

    script_dir = Path(__file__).parent
    csv_file = Path(data_file)
    if not csv_file.is_absolute():
        csv_file = script_dir / csv_file

    if not csv_file.exists():
        print(f"❌ Error: File not found: {csv_file}")
        return None, None, None

    file_base_name = csv_file.stem
    # Delegate loading, delimiter guessing, and duplicate handling to csv_utils
    df = load_and_prepare_csv(csv_file, text_column, delimiter)
    print(f"Loaded: {csv_file} | Shape: {df.shape}")

    if max_samples and max_samples > 0:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples for testing")

    print(f"\nProcessing {len(df)} documents for classification...\n")
    print("=" * 60)
    print("CLASSIFYING DOCUMENTS")
    print("=" * 60)

    output_dir = script_dir / "out-3"
    output_dir.mkdir(parents=True, exist_ok=True)

    classified_df = df.copy()
    classified_df["Category"] = classified_df[text_column].fillna("").apply(classify_prompt)

    print("Detecting languages...")
    classified_df["detected_language"] = classified_df[text_column].apply(detect_language)
    output_file = str(output_dir / f"{file_base_name}_classified.xlsx")
    classified_df.to_excel(output_file, index=False, sheet_name="Classifications")

    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    category_summary = classified_df['Category'].value_counts().reset_index()
    category_summary.columns = ['Category', 'Count']
    selected_category_codes = get_selected_category_codes()
    # Record which categories were active when generating this summary
    category_summary["SelectedCategories"] = ", ".join(selected_category_codes)
    summary_file = str(output_dir / f"{file_base_name}_summary.xlsx")
    category_summary.to_excel(summary_file, index=False, sheet_name='Summary')

    print("\nCategory Distribution:")
    print(category_summary.to_string(index=False))

    print(f"\nAnalysis complete!")
    print(f"Classified {len(classified_df)} documents")
    print(f"Unique categories: {classified_df['Category'].nunique()}")
    print(f"Results saved to {output_dir}")

    return classified_df, category_summary, df


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    parser = parse_args()
    args = parser.parse_args()

    if args.data_files:
        files_to_process = [
            resolve_data_file(script_dir, path_str) for path_str in args.data_files
        ]
    else:
        candidates = discover_csv_files(script_dir)
        if not candidates:
            print("No data directories with CSV files were found.")
            exit(1)
        files_to_process = prompt_user_for_files(candidates, script_dir)

    for file_path in files_to_process:
        print(f"\nProcessing file: {format_display_path(file_path, script_dir)}\n")
        main(
            data_file=file_path,
            text_column=args.text_column,
            delimiter=args.delimiter,
            max_samples=args.max_samples,
        )
