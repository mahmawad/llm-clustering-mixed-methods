# LLM Clustering Mixed Methods Project

## 🎯 Project Overview

This project implements a comprehensive AI-powered text clustering pipeline for analyzing mixed-methods survey responses using Topic Modeling with BERTopic, Sentence Transformers, and Large Language Models (LLMs). It automates the analysis of open-ended textual responses to identify patterns and categories efficiently.

## 📋 Workflow Pipeline

### 1. Data Input & Loading (`csv_utils.py`)

```
CSV file → Duplicate Check → Clean DataFrame
```

- **Input**: Raw CSV files with textual survey responses
- **Features**:
  - Automatic encoding detection (UTF-8, Latin-1, CP1252, ISO-8859-1)
  - Duplicate row identification and removal
  - Data validation and statistics
- **Output**: Clean pandas DataFrame ready for analysis

### 2. Topic Modeling & Clustering (`topic_modeling.py`)

#### Embedding Generation

- **Model**: Sentence-BERT (`all-MiniLM-L6-v2`)
- **Framework**: Sentence-Transformers
- **Hardware**: CPU/GPU optimized
- **Output**: High-dimensional document embeddings

#### Dimensionality Reduction

- **Algorithm**: UMAP
- **Parameters**: 15 neighbors, 5 components, cosine distance
- **Purpose**: Reduce embedding dimensions for clustering and visualization

#### Clustering

- **Algorithm**: K-Means Clustering
- **Approach**: Configurable cluster numbers based on Data scale
- **Robustness**: Outlier detection and reduction

#### Topic Representation

- **Model**: OpenAI GPT-3.5-Turbo
- **Purpose**: Generate human-readable topic labels/descriptions
- **Method**: C-TF-IDF + LLM-based fine-tuning

### 3. Topic Refinement

- Outlier detection and reduction using embedding-based strategy
- Automatic topic label generation and updating
- Topic frequency analysis

### 4. Visualization & Analysis (`visualize_topics()`)

```
Topics → Interactive HTML Dashboard
```

- **Output**: Interactive Plotly-based HTML visualization
- **Features**:
  - Document-topic scatter plot
  - Hover information with document content
  - Color-coded by cluster assignment
  - Saved to: `out-2/documents_visualization.html`

## 🏗️ Project Structure

```
llm-clustering-mixed-methods/
├── main.py                      # Main execution pipeline
├── csv_utils.py                 # CSV loading & preprocessing
├── topic_modeling.py            # Topic modeling functions
├── config/
│   └── config.toml             # API keys & configuration (git-ignored)
├── data-2/
│   └── 1prompts_v2.csv         # Input survey responses
├── out-2/
│   └── documents_visualization.html  # Generated visualization
└── requirements.txt            # Python dependencies
```

## 🔧 Technical Stack

| Component          | Tools                                    | Purpose                       |
| ------------------ | ---------------------------------------- | ----------------------------- |
| **Data Loading**   | pandas, csv_utils                        | CSV processing & validation   |
| **Embeddings**     | Sentence-Transformers (all-MiniLM-L6-v2) | Document embedding generation |
| **Dimensionality** | UMAP                                     | Embedding reduction (5D → 2D) |
| **Clustering**     | scikit-learn (K-Means)                   | Document grouping             |
| **Topic Labels**   | OpenAI API (GPT-3.5-Turbo)               | LLM-based topic generation    |
| **Topic Modeling** | BERTopic                                 | End-to-end topic modeling     |
| **Visualization**  | Plotly                                   | Interactive HTML dashboards   |

## 📊 Current Implementation Status

### ✅ Completed Features

1. **CSV Data Import & Preprocessing**

   - Automatic encoding detection
   - Duplicate detection and removal
   - Data quality checks

2. **Topic Modeling with BERTopic**

   - Sentence-Bert embeddings
   - K-Means clustering
   - UMAP dimensionality reduction
   - GPT-3.5 topic labeling

3. **Visualization**
   - Interactive HTML document visualization
   - Topic distribution analysis
   - Topic frequency statistics

### 🔄 In Progress / Future

- [ ] Comparison with predefined categories/gold standard
- [x] Export clustering results as xlsx file
- [x] Multi-language support (German/English)

## 🚀 Usage

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create/update `config/config.toml`:

```toml
[api]
openai_api_key = "your-api-key-here"
```

### Running the Pipeline

```bash
python main.py
```

**Output**:

- Console logs with processing progress
- Interactive visualization: `out-2/documents_visualization.html`
- Identified topics and cluster assignments

## 📚 Dependencies

- `pandas` - Data manipulation
- `scikit-learn` - K-Means clustering
- `sentence-transformers` - Embedding models
- `umap-learn` - Dimensionality reduction
- `bertopic` - Topic modeling framework
- `openai` - GPT API integration
- `plotly` - Interactive visualizations
- `toml` - Configuration file parsing

## 📄 License & Attribution

### BERTopic

This project uses **BERTopic** for topic modeling.

**Citation:**

```
@inproceedings{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  booktitle={arXiv preprint arXiv:2203.05556},
  year={2022}
}
```

**License**: MIT License  
**Project**: https://github.com/MaartenGr/BERTopic

## 📖 Scientific Context: Mixed-Methods Research

In Mixed-Methods studies, large volumes of open-ended text responses are frequently collected. Manual analysis is time-consuming and error-prone. This project demonstrates how modern AI models (LLMs + embeddings) can automatically:

- **Identify patterns** in qualitative text data
- **Discover categories** automatically from responses
- **Reduce analysis time** through automation
- **Maintain rigor** through systematic clustering
- **Integrate with** quantitative analysis pipelines

### Advantages for Mixed-Methods:

✅ Scalable analysis of large text volumes  
✅ Objective, reproducible categorization  
✅ Integration with quantitative metrics  
✅ Hypothesis generation from exploratory clustering  
✅ Bridge between qualitative insights and quantitative analysis

### Limitations:

⚠️ Requires validation against human judgment  
⚠️ Quality depends on LLM model selection  
⚠️ May lose nuanced context in large-scale clustering  
⚠️ Requires careful prompt engineering for topic labels

---

**Last Updated**: December 2025
