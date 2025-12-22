"""Analyze which labels are most wrongly classified."""

import pandas as pd

# Load confusion matrix
cm_df = pd.read_csv('out-2/confusion_matrix.csv', index_col=0)

print('='*70)
print('MISCLASSIFICATION ANALYSIS - Which labels are wrong most often?')
print('='*70)

error_rates = []

for true_label in cm_df.index:
    total_actual = cm_df.loc[true_label].sum()
    
    if total_actual == 0:
        continue
    
    correct = cm_df.loc[true_label, true_label]
    incorrect = total_actual - correct
    error_rate = incorrect / total_actual * 100
    
    error_rates.append({
        'Label': true_label,
        'Total_Samples': int(total_actual),
        'Correct': int(correct),
        'Wrong': int(incorrect),
        'Error_Rate_%': round(error_rate, 1)
    })

# Convert to DataFrame and sort by error rate
error_df = pd.DataFrame(error_rates).sort_values('Error_Rate_%', ascending=False)

print("\n" + error_df.to_string(index=False))

print('\n' + '='*70)
print('TOP 3 WORST PERFORMING LABELS (Most Misclassifications):')
print('='*70)

for i, row in error_df.head(3).iterrows():
    label = row['Label']
    error_rate = row['Error_Rate_%']
    wrong = int(row['Wrong'])
    total = int(row['Total_Samples'])
    
    print(f"\n{label}: {error_rate}% error rate ({wrong}/{total} misclassified)")
    
    # Show what it was misclassified as
    misclass = cm_df.loc[label].drop(label)
    misclass = misclass[misclass > 0].sort_values(ascending=False)
    
    if len(misclass) > 0:
        print("  Misclassified as:")
        for predicted, count in misclass.items():
            pct = count / total * 100
            print(f"    {predicted}: {int(count)} samples ({pct:.1f}%)")
    else:
        print("  No misclassifications recorded")

print('\n' + '='*70)
print('LABELS NEVER PREDICTED (False Negatives):')
print('='*70)

# Find labels that were never predicted
predicted_labels = [col for col in cm_df.columns]
print(f"Predicted labels: {predicted_labels}")
print(f"True labels: {cm_df.index.tolist()}")

never_predicted = set(cm_df.index) - set(predicted_labels)
if never_predicted:
    for label in never_predicted:
        total = cm_df.loc[label].sum()
        if total > 0:
            print(f"\n{label}: {int(total)} actual samples but NEVER predicted correctly")
            misclass = cm_df.loc[label][cm_df.loc[label] > 0]
            for predicted, count in misclass.items():
                pct = count / total * 100
                print(f"  → Misclassified as {predicted}: {int(count)} ({pct:.1f}%)")
else:
    print("All true labels were predicted at least once")
