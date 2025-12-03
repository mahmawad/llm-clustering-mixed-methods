import os
from pathlib import Path
from csv_utils import analyze_csv, remove_duplicates
from topic_modeling import get_topic_modeling, get_final_tm, visualize_topics


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
    
    print(f"\nProcessing {len(df)} documents for topic modeling...\n")
    
    # Train topic model
    print("=" * 60)
    print("TRAINING TOPIC MODEL")
    print("=" * 60)
    
    cluster_nr = 5
    topic_model, topics, probs, embeddings = get_topic_modeling(df, cluster_nr)
    
    # Refine topic model and get final results
    print("\n" + "=" * 60)
    print("REFINING TOPIC MODEL")
    print("=" * 60)
    
    topic_freq = get_final_tm(topic_model, topics, probs, df)
    
    # Visualize topics and documents
    print("\n" + "=" * 60)
    print("VISUALIZING RESULTS")
    print("=" * 60)
    
    docs = df[text_column].tolist()
    output_file = script_dir / "out-2" / "documents_visualization.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fig = visualize_topics(topic_model, docs, reduced_embeddings=None, output_file=str(output_file))
    
    print(f"\nAnalysis complete!")
    print(f"Found {len(topic_freq)} topics")
    print(f"Visualization saved to {output_file}")
    
    return topic_model, topics, probs, df, topic_freq


if __name__ == "__main__":
    topic_model, topics, probs, df, topic_freq = main()
