# LLM Clustering Mixed Methods Project

## 🎯 Project Overview

This project implements a comprehensive text clustering pipeline for analyzing survey responses and textual data using Large Language Models (LLMs)

## 📋 Workflow Pipeline

### 1. Data Input & Loading

```
CSV file  → DataFrame
```

- **Input**: Raw CSV files with textual data
- **Tools**: `csv_utils.py` for loading and duplicate checking
- **Output**: Clean pandas DataFrame

### 2. Text Preprocessing

```
Raw Text → Cleaned & Normalized Text
```

- **Remove duplicates**: Identify and eliminate duplicate entries
- **Optional**: if needed
- **Text cleaning**: Remove punctuation, special characters, URLs
- **Stopword removal**: Using NLTK German/English stopword lists
- **Stemming/Lemmatization**: Optional word normalization

### 3. Text Embeddings Generation

**Option A: LLM Embeddings**

- OpenAI text-embedding-3-small/large
- Sentence-BERT models

### 4. Clustering Analysis

```
Embeddings → Cluster Labels
```

**Algorithms:**

- **K-Means**: For spherical clusters
- **HDBSCAN**: For density-based clustering

### 5. Visualization & Analysis

```
Clusters → Interactive Plots & Reports
```

- **2D/3D scatter plots** with dimensionality reduction ( UMAP)

## 📊 Expected Outputs

### Preprocessing Results

### Clustering Results

## 🔧 Technical Stack

| Component         | Tools                             | Purpose                |
| ----------------- | --------------------------------- | ---------------------- |
| **Data Loading**  | pandas, csv_utils                 | CSV processing         |
| **Preprocessing** | NLTK, scikit-learn                | Text cleaning          |
| **Embeddings**    | OpenAI API, sentence-transformers | Vector representations |
| **Clustering**    | scikit-learn, scipy               | Grouping algorithms    |
| **Visualization** | matplotlib, plotly, seaborn       | Data visualization     |

## 🔄 Next Steps

- [ ] Implement embedding generation module
- [ ] Add clustering algorithms implementation
- [ ] Create visualization dashboard
