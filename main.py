from pathlib import Path

from csv_utils import analyze_csv, remove_duplicates
from helper import classify_documents


def main():
    # Load and analyze CSV data
    print("=" * 60)
    print("LOADING AND ANALYZING DATA")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    csv_file = script_dir / "data-2" / "1prompts_v2.csv"
    text_column = "Prompt"
    
    # Analyze CSV for duplicates
    df, dup_info = analyze_csv(str(csv_file), text_column=text_column)
    
    # Remove duplicates if any exist
    if dup_info['duplicate_rows'] > 0:
        df = remove_duplicates(df, columns=[text_column])
    
    print(f"\nProcessing {len(df)} documents for classification...\n")
    
    # Classify documents using predefined categories
    print("=" * 60)
    print("CLASSIFYING DOCUMENTS")
    print("=" * 60)
    
    output_dir = script_dir / "out-2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classified_df = classify_documents(
        df,
        text_column=text_column,
        output_file=str(output_dir / "classified_documents.xlsx"),
    )
    
    # Create a summary report
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    category_summary = classified_df['Category'].value_counts().reset_index()
    category_summary.columns = ['Category', 'Count']
    category_summary.to_excel(str(output_dir / "category_summary.xlsx"), index=False, sheet_name='Summary')
    
    print("\nCategory Distribution:")
    print(category_summary.to_string(index=False))
    
    print(f"\nAnalysis complete!")
    print(f"Classified {len(classified_df)} documents")
    print(f"Unique categories: {classified_df['Category'].nunique()}")
    print(f"Results saved to {output_dir}")
    
    return classified_df, category_summary, df


if __name__ == "__main__":
    classified_df, category_summary, df = main()
