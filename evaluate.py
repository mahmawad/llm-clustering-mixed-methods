"""Evaluate classification output against ground truth using confusion matrix."""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_ground_truth(filepath):
    """Load ground truth from the reference CSV file."""
    # The file uses semicolon delimiter and has a specific format
    df = pd.read_csv(filepath, sep=";", header=None)
    
    # Extract entryId (column 0) and category (column 1)
    ground_truth = {}
    for idx, row in df.iterrows():
        if pd.notna(row[0]) and pd.notna(row[1]) and row[0] != 'entryId':
            try:
                entry_id = int(row[0])
                category = row[1].strip().strip('"')
                ground_truth[entry_id] = category
            except (ValueError, AttributeError):
                continue
    
    return ground_truth


def load_predictions(filepath):
    """Load predictions from the classified_documents.xlsx file."""
    df = pd.read_excel(filepath)
    
    # The file has entryId and Category columns
    predictions = {}
    if 'entryId' in df.columns and 'Category' in df.columns:
        for idx, row in df.iterrows():
            entry_id = row['entryId']
            category = row['Category']
            predictions[entry_id] = category
    
    return predictions, df


def create_confusion_matrix(ground_truth, predictions):
    """Create and visualize confusion matrix."""
    
    # Get all unique categories
    all_categories = sorted(set(list(ground_truth.values()) + list(predictions.values())))
    
    # Match predictions to ground truth
    matched_truth = []
    matched_pred = []
    
    for entry_id, truth_label in ground_truth.items():
        if entry_id in predictions:
            matched_truth.append(truth_label)
            matched_pred.append(predictions[entry_id])
    
    if not matched_truth:
        print("⚠️  No matching entries found between ground truth and predictions!")
        return None, None, 0
    
    # Create confusion matrix
    cm = confusion_matrix(matched_truth, matched_pred, labels=all_categories)
    
    # Calculate accuracy
    accuracy = accuracy_score(matched_truth, matched_pred)
    
    # Print classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(matched_truth, matched_pred, labels=all_categories))
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Matched entries: {len(matched_truth)}")
    
    return cm, all_categories, accuracy


def plot_confusion_matrix(cm, categories, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix: Ground Truth vs Predictions')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_path}")
    plt.close()


def main():
    script_dir = Path(__file__).parent
    
    # File paths
    ground_truth_file = script_dir / "out-2" / "2prompts_v2KI-gpt4omini.csv"
    predictions_file = script_dir / "out-2" / "o4_mini_classified_documents.xlsx"
    output_plot = script_dir / "out-2" / "o4_mini_confusion_matrix.png"
    
    print("=" * 70)
    print("CONFUSION MATRIX EVALUATION")
    print("=" * 70)
    
    # Check if files exist
    if not ground_truth_file.exists():
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        return
    
    if not predictions_file.exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        return
    
    print(f"\nLoading ground truth from: {ground_truth_file}")
    ground_truth = load_ground_truth(str(ground_truth_file))
    print(f"✓ Loaded {len(ground_truth)} ground truth labels")
    
    print(f"\nLoading predictions from: {predictions_file}")
    predictions, df = load_predictions(str(predictions_file))
    print(f"✓ Loaded {len(predictions)} predictions")
    
    print(f"\nGround truth categories: {sorted(set(ground_truth.values()))}")
    print(f"Predicted categories: {sorted(set(predictions.values()))}")
    
    # Create confusion matrix
    cm, categories, accuracy = create_confusion_matrix(ground_truth, predictions)
    
    if cm is not None:
        # Plot confusion matrix
        plot_confusion_matrix(cm, categories, str(output_plot))
        
        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(cm, index=categories, columns=categories)
        cm_df.to_csv(script_dir / "out-2" / "o4_mini_confusion_matrix.csv")
        print(f"✓ Confusion matrix CSV saved to: {script_dir / 'out-2' / 'o4_mini_confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
