# LLM Text Classification Project

## 🎯 Project Overview

This project implements a comprehensive AI-powered text classification pipeline for analyzing mixed-methods survey responses. It uses predefined category schemes and Large Language Models (LLMs) to automatically classify open-ended textual responses with high precision. The system combines language detection with semantic classification to support multilingual datasets.

## 📋 Workflow Pipeline

### 1. Data Input & Loading (`csv_utils.py`)

```
CSV file → Delimiter Detection → Duplicate Removal → Clean DataFrame
```

- **Input**: Raw CSV files with textual responses
- **Functions**:
  - `csv_to_df()`: Loads CSV with automatic encoding detection (UTF-8, Latin-1, CP1252, ISO-8859-1)
  - `check_duplicates()`: Identifies duplicate rows with detailed statistics
  - `remove_duplicates()`: Removes redundant entries while preserving data integrity
  - `load_csv_with_fallback()`: Handles various delimiter formats (comma, semicolon, tab, pipe)
  - `load_and_prepare_csv()`: Orchestrates loading, deduplication, and preparation
- **Output**: Clean pandas DataFrame with no duplicates, ready for classification

### 2. Language Detection (`llm_helper.py`)

```
Text → FastText Language Model → Language Code
```

- **Function**: `detect_language(text)`
- **Model**: FastText (`lid.176.ftz`) - 176 language identification model
- **Output**: Language code (e.g., 'en', 'de', 'fr') with confidence score
- **Purpose**: Supports multilingual analysis and downstream language-specific processing

### 3. Document Classification (`llm_helper.py`)

```
User Query → LLM Classification → Category Code
```

#### Category Schema

The system uses a hierarchical category structure organized into **5 main groups**:

| Group          | Categories            | Purpose                                                    |
| -------------- | --------------------- | ---------------------------------------------------------- |
| **Defining**   | D.I, D.G              | Problem identification and goal setting                    |
| **Seeking**    | S.S, S.SL, S.EQ       | Information retrieval and quality evaluation               |
| **Engaging**   | E.RV, E.O, E.RF, E.RH | Content review, organization, transformation, and practice |
| **Reflecting** | R.ET, R.ES            | Task evaluation and self-reflection                        |
| **Other**      | OTHER                 | Off-topic or unclassifiable queries                        |

#### Classification Functions

- **`get_active_category_codes()`**: Returns cached category selection; prompts user interactively on first call
- **`build_category_section(selected_codes)`**: Formats selected categories into LLM-friendly prompt section
- **`classify_prompt(doc, model)`**:
  - Classifies a single document into predefined categories
  - Uses OpenAI GPT-4o-Mini for semantic understanding
  - Parameters: `temperature=0` (deterministic), `max_tokens=10` (short response)
  - Returns: Category code or "ERROR" on failure

#### Configuration

- **Model**: OpenAI GPT-4o-Mini
- **Temperature**: 0 (deterministic, reproducible results)
- **API Key**: Loaded from `config/config.toml`
- **Category Selection**: User can choose which categories to include in classification

### 4. Results Aggregation & Export

```
Classifications → Summary Statistics → Excel Output
```

- **Output Files**:
  - `{filename}_classified.xlsx`: Full dataset with assigned categories and detected languages
  - `{filename}_summary.xlsx`: Category distribution summary with counts
- **Functions**:
  - Main pipeline generates classification results
  - Results exported to `out-3/` directory

## 🏗️ Project Structure

```
llm-clustering-mixed-methods/
├── main.py                             # Main execution pipeline
├── csv_utils.py                        # CSV loading, preprocessing & file management
├── llm_helper.py                       # LLM classification & language detection
├── evaluate.py                         # Evaluation with confusion matrices
├── analyze_errors.py                   # Error analysis & misclassification insights
├── config/
│   └── config.toml                    # API keys & configuration (git-ignored)
├── data-2/
│   └── 1prompts_v2.csv               # Example input data
├── data-3/
│   └── *.csv                         # Additional input datasets
├── out-3/
│   ├── {filename}_classified.xlsx     # Full classification results
│   └── {filename}_summary.xlsx        # Category distribution summary
├── lid.176.ftz                        # FastText language identification model
└── requirements.txt                   # Python dependencies
```

## 🔧 Technical Stack

| Component              | Tools                    | Purpose                              |
| ---------------------- | ------------------------ | ------------------------------------ |
| **Data Loading**       | pandas, csv_utils        | CSV processing & validation          |
| **Classification**     | OpenAI API (GPT-4o-Mini) | LLM-based text classification        |
| **Language Detection** | FastText (lid.176.ftz)   | Multilingual language identification |
| **Evaluation**         | scikit-learn             | Confusion matrices & metrics         |
| **Export**             | openpyxl, pandas         | Excel output generation              |
| **Configuration**      | toml                     | API key & settings management        |

## 📊 Current Implementation Status

### ✅ Completed Features

1. **CSV Data Import & Preprocessing**

   - Automatic encoding detection
   - Duplicate detection and removal
   - Flexible delimiter handling (auto-detection)
   - Interactive file selection from multiple data directories

2. **Document Classification**

   - GPT-4o-Mini powered LLM classification
   - Predefined hierarchical category schema (12 categories + OTHER)
   - User-configurable category selection
   - Interactive category menu on first run
   - Deterministic classification (temperature=0)

3. **Language Detection**

   - FastText-based multilingual language identification
   - Support for 176 languages
   - Confidence scoring
   - Integrated with classification pipeline

4. **Results Export**

   - Full classified dataset to Excel (with detected languages)
   - Category distribution summary statistics
   - Multiple file processing support
   - Organized output to `out-3/` directory

5. **Evaluation & Analysis**
   - Confusion matrix generation
   - Classification accuracy metrics
   - Misclassification analysis
   - Error rate reporting per category
   - False negative detection

### 🔄 Features

- [x] Interactive file selection from multiple data directories
- [x] CSV data loading with encoding detection
- [x] Duplicate detection and removal
- [x] LLM-based document classification
- [x] Language detection for multilingual support
- [x] Export classification results to Excel
- [x] Category distribution analysis
- [x] Evaluation and confusion matrix generation
- [x] Error analysis and insights

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

#### Option 1: Interactive Mode (Default)

```bash
python main.py
```

The system will:

1. Discover CSV files in `data-*/` directories
2. Present a numbered list of available files
3. Prompt you to select files to process (enter numbers, e.g., `1 3` or `all`)
4. Ask which classification categories to include (or use all by default)
5. Classify all documents in the selected files
6. Generate Excel outputs with results

#### Option 2: Command-Line Arguments

```bash
python main.py --data-files data-2/1prompts_v2.csv data-3/other_data.csv
```

**Available Arguments:**

- `--data-files FILE [FILE ...]`: Specify input CSV files (relative or absolute paths)
- `--text-column COLUMN`: Column name containing text to classify (default: `content`)
- `--delimiter DELIMITER`: CSV delimiter character (default: `,`)
- `--max-samples N`: Limit processing to first N samples (useful for testing)

**Example:**

```bash
python main.py -f data-2/1prompts_v2.csv --text-column Prompt --max-samples 50
```

### Output Files

All results are saved to `out-3/` directory:

- **`{filename}_classified.xlsx`**:

  - Full dataset with classifications
  - Columns: Original data + `Category` + `detected_language`
  - One row per document

- **`{filename}_summary.xlsx`**:
  - Category distribution summary
  - Columns: `Category`, `Count`, `SelectedCategories`
  - Helpful for understanding classification breakdown

### Evaluation & Error Analysis

After classification, evaluate against ground truth labels:

```bash
python evaluate.py
python analyze_errors.py
```

**Output**:

- Confusion matrices (CSV format)
- Classification accuracy metrics
- Error rate analysis per category
- Misclassification patterns

## 📚 Dependencies

- `pandas` - Data manipulation and CSV handling
- `openai` - GPT-4o-Mini API integration for classification
- `fasttext` - Language identification model
- `scikit-learn` - Confusion matrices and evaluation metrics
- `openpyxl` - Excel file generation and manipulation
- `toml` - Configuration file parsing

## 📖 API Reference

### csv_utils.py

#### `csv_to_df(file_path, delimiter=';', encoding='utf-8')`

Loads a CSV file with automatic encoding detection.

- **Parameters**:
  - `file_path`: Path to CSV file
  - `delimiter`: Field separator (default: semicolon)
  - `encoding`: Preferred encoding (tries multiple if fails)
- **Returns**: pandas DataFrame

#### `check_duplicates(df, columns=None)`

Identifies duplicate rows in the dataset.

- **Parameters**:
  - `df`: Input DataFrame
  - `columns`: Specific columns to check (None = all columns)
- **Returns**: Dictionary with duplicate statistics

#### `remove_duplicates(df, columns=None, keep='first')`

Removes duplicate rows from DataFrame.

- **Parameters**:
  - `df`: Input DataFrame
  - `columns`: Columns to consider for deduplication
  - `keep`: Which duplicate to keep ('first', 'last', False)
- **Returns**: Deduplicated DataFrame

#### `load_and_prepare_csv(csv_file, text_column, delimiter=',')`

Complete CSV loading and preparation pipeline.

- **Parameters**:
  - `csv_file`: Path to CSV file
  - `text_column`: Column name for classification
  - `delimiter`: Field separator
- **Returns**: Clean, deduplicated DataFrame

#### `discover_csv_files(script_dir)`

Finds all CSV files in `data-*` subdirectories.

- **Parameters**:
  - `script_dir`: Project root directory
- **Returns**: List of CSV file paths

#### `prompt_user_for_files(candidates, script_dir)`

Interactive menu for file selection.

- **Parameters**:
  - `candidates`: List of available CSV files
  - `script_dir`: Project root for display paths
- **Returns**: List of selected file paths

### llm_helper.py

#### `detect_language(text)`

Detects language of input text using FastText model.

- **Parameters**:
  - `text`: Input text string
- **Returns**: Language code (e.g., 'en', 'de', 'fr')

#### `get_active_category_codes()`

Returns currently active classification categories (cached).

- **Parameters**: None
- **Returns**: List of category codes (e.g., ['D.I', 'D.G', 'S.S', ...])

#### `build_category_section(selected_codes)`

Formats category definitions for LLM prompt.

- **Parameters**:
  - `selected_codes`: List of category codes to include
- **Returns**: Formatted string with category descriptions

#### `classify_prompt(doc, model='gpt-4o-mini')`

Classifies a single document into a predefined category.

- **Parameters**:
  - `doc`: Text to classify
  - `model`: OpenAI model name (default: gpt-4o-mini)
- **Returns**: Category code (e.g., 'S.S', 'E.RV', 'OTHER')
- **Error Handling**: Returns "ERROR" if classification fails

#### `get_selected_category_codes()`

Returns copy of cached category selection.

- **Parameters**: None
- **Returns**: List of selected category codes

#### Category Constants

**CATEGORY_METADATA**: Dictionary with category definitions

- Each entry contains: `title`, `description`
- Used for prompt generation and user menu

**CATEGORY_GROUPS**: Hierarchical grouping of categories

- Defining: D.I, D.G
- Seeking: S.S, S.SL, S.EQ
- Engaging: E.RV, E.O, E.RF, E.RH
- Reflecting: R.ET, R.ES
- Other: OTHER

### main.py

#### `main(data_file, text_column='content', delimiter=',', max_samples=None)`

Main classification pipeline function.

- **Parameters**:
  - `data_file`: Path to input CSV
  - `text_column`: Column containing text to classify
  - `delimiter`: CSV field separator
  - `max_samples`: Limit processing (None = all)
- **Returns**: Tuple of (classified_df, summary_df, original_df)
- **Output Files**:
  - `out-3/{filename}_classified.xlsx`: Full results
  - `out-3/{filename}_summary.xlsx`: Category summary

### evaluate.py

#### `load_ground_truth(filepath)`

Loads reference categories for evaluation.

- **Parameters**:
  - `filepath`: Path to ground truth reference file
- **Returns**: Dictionary mapping entry IDs to expected categories

#### `load_predictions(filepath)`

Loads classification results from Excel file.

- **Parameters**:
  - `filepath`: Path to classified results Excel file
- **Returns**: Tuple of (predictions dict, DataFrame)

#### `create_confusion_matrix(ground_truth, predictions)`

Generates confusion matrix and visualizations.

- **Parameters**:
  - `ground_truth`: Dictionary of expected categories
  - `predictions`: Dictionary of predicted categories
- **Returns**: Confusion matrix DataFrame

## � Scientific Context: Mixed-Methods Research

In Mixed-Methods studies, large volumes of open-ended text responses are frequently collected. Manual analysis is time-consuming and error-prone. This project demonstrates how modern AI models (LLMs) can automatically:

- **Classify responses** into predefined analytical categories
- **Identify patterns** in qualitative text data
- **Standardize coding** with consistent, reproducible categorization
- **Integrate with** quantitative analysis pipelines
- **Support multilingual** research with automatic language detection

### Advantages for Mixed-Methods:

✅ **Scalability**: Processes large volumes of text efficiently  
✅ **Consistency**: Deterministic classification (temperature=0) ensures reproducibility  
✅ **Objectivity**: LLM-based coding reduces analyst bias  
✅ **Validation**: Built-in evaluation tools for inter-rater agreement simulation  
✅ **Multilingual**: Supports responses in 176+ languages  
✅ **Integration**: Exports to Excel for further quantitative analysis

### Limitations:

⚠️ **Validation Required**: Predictions should be validated against human judgment  
⚠️ **Model Dependent**: Quality depends on selected LLM and category definitions  
⚠️ **Nuance Loss**: May oversimplify complex or ambiguous responses  
⚠️ **Category Design**: Requires careful category definition and prompt engineering  
⚠️ **Cost**: API-based classification has per-token costs

## 📖 Technical Details

### Classification Strategy

The system uses a **prompt-based classification approach**:

1. **Category Selection**: User selects which categories to include (or uses all)
2. **Prompt Building**: Only selected categories are included in the LLM prompt
3. **Single Document**: Each document is classified independently
4. **Deterministic**: Temperature=0 ensures consistent results
5. **Error Handling**: Failed classifications return "ERROR" for later review

### Language Detection Integration

- **Automatic Detection**: Languages detected for all classified documents
- **Output**: Language codes stored alongside classifications
- **Use Case**: Enables downstream language-specific analysis or filtering

### Data Quality

- **Deduplication**: Removes redundant rows before classification (saves API costs)
- **Encoding Handling**: Automatically detects and handles multiple character encodings
- **Delimiter Flexibility**: Handles various CSV formats (comma, semicolon, tab, pipe)

## 📄 License & Attribution

### Dependencies

- **FastText** (Language Identification): MIT License
- **OpenAI API**: Commercial use (requires API key)
- **scikit-learn**: BSD License
- **pandas**: BSD License

### Citation

If you use this project in academic work, please cite:

```bibtex
@software{llm_classification_2025,
  title={LLM Text Classification for Mixed-Methods Research},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/llm-classification}
}
```

---

**Last Updated**: December 2025  
**Status**: Active Development
