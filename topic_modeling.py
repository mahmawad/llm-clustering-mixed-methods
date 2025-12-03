from sentence_transformers import SentenceTransformer
from bertopic.representation import OpenAI
from openai import OpenAI as OpenAIClient
from umap import UMAP
from sklearn.cluster import KMeans
from bertopic import BERTopic
import torch
import toml
from pathlib import Path

# Force CPU usage
torch.cuda.is_available = lambda: False

# Initialize embedding model on CPU
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load API key from config
config_path = Path(__file__).parent / "config" / "config.toml"
config = toml.load(config_path)
api_key = config["api"]["openai_api_key"]

# Create the OpenAI API client
client = OpenAIClient(api_key=api_key)


def get_topic_modeling(df, cluster_nr):
    """
    Generates a topic model for a given DataFrame using various NLP and clustering techniques.

    Args:
        df (pd.DataFrame): DataFrame containing the preprocessed text data.
        cluster_nr (int): number of desired generated clusters.

    Returns:
        topic_model: The trained BERTopic model.
        topics: The topics identified by the model.
        probs: The probabilities of the topics.
        embeddings: Document embeddings for visualization.
    """
    # Encode documents using sentence transformer
    embeddings = embedding_model.encode(df['Prompt'].tolist(), show_progress_bar=True)
    
    # Initialize UMAP model for dimensionality reduction (for visualization)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    # Initialize KMeans clustering model
    cluster_model = KMeans(random_state=42, n_clusters=cluster_nr)
    
    # Initialize text generation model with OpenAI
    representation_model = OpenAI(client, model="gpt-3.5-turbo", delay_in_seconds=5)
    
    # Initialize and train BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=True,
        representation_model=representation_model,
        verbose=True,
        nr_topics=cluster_nr
    )
    
    # Fit the topic model and transform the data
    topics, probs = topic_model.fit_transform(df['Prompt'].values)
    
    return topic_model, topics, probs, embeddings


def get_final_tm(topic_model, topics, probs, df):
    """
    Refines the topic model by reducing outliers and updating topic labels.

    Args:
        topic_model: The trained BERTopic model.
        topics: The topics identified by the model.
        probs: The probabilities of the topics.
        df (pd.DataFrame): DataFrame containing the original text data and metadata.

    Returns:
        pd.DataFrame: Topic frequency information.
    """
    # Check if there are outliers and reduce using the BERTopic model
    if -1 in topics:
        new_topics = topic_model.reduce_outliers(df['Prompt'].values, topics, probabilities=probs, strategy="embeddings")
        topic_model.update_topics(df['Prompt'].values, topics=new_topics)
    
    # Get topic frequency
    topic_freq = topic_model.get_topic_freq()
    return topic_freq


def visualize_topics(topic_model, docs, reduced_embeddings=None, output_file="out-2/documents_visualization.html"):
    """
    Visualize documents and topics using BERTopic.

    Args:
        topic_model: The trained BERTopic model.
        docs: List of documents.
        reduced_embeddings: Optional pre-computed 2D embeddings for visualization.
        output_file: Path to save the HTML visualization.

    Returns:
        fig: Plotly figure object.
    """
    if reduced_embeddings is not None:
        fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    else:
        fig = topic_model.visualize_documents(docs)
    
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    return fig