# Sentiment Analysis on IMDB Movie Reviews

## Overview

This project performs sentiment analysis on the IMDB Large Movie Review Dataset. The goal is to classify movie reviews as either positive or negative using machine learning techniques. The project includes data extraction, preprocessing, and preparation for model training and evaluation.

## Setup & Project Commands

Before running any scripts, it is recommended to create a Python virtual environment to isolate dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### Install Dependencies

You can install all required dependencies using either `pip` or `poetry`:

- **With pip:**
  ```bash
  pip install -r requirements.txt
  ```
- **With poetry:**
  ```bash
  poetry install
  ```

### Run Scripts

- **Extract and aggregate IMDB reviews:**
  ```bash
  poetry run python src/pipeline-utils.py
  ```
- **Jupyter Notebook for exploration:**
  ```bash
  poetry run jupyter notebook src/sentinment-analysis.ipynb
  ```

### Run Tests

- **With pytest (recommended):**
  ```bash
  poetry run pytest
  ```
- **With Makefile (if available):**
  ```bash
  make test
  ```

### Run Lint

- **With flake8:**
  ```bash
  poetry run flake8 src
  ```
- **With Makefile (if available):**
  ```bash
  make lint
  ```

### Install a New Dependency

- **With pip:**
  ```bash
  pip install <package-name>
  pip freeze > requirements.txt  # Update requirements.txt
  ```
- **With poetry:**
  ```bash
  poetry add <package-name>
  ```

A `Makefile` is provided for convenience. If you have `make` installed, you can use `make lint` and `make test` as shortcuts for linting and testing.

Refer to this section whenever you need to set up, test, lint, or extend the project.

## Dataset

The dataset used is the [Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 labeled reviews (25,000 for training and 25,000 for testing) and 50,000 unlabeled reviews for unsupervised learning. Reviews are split into positive and negative categories, and each review is stored as a text file with its sentiment label and rating encoded in the filename.

- **Source:** See `raw_data/aclImdb/README` for full dataset details and citation.
- **Labels:**
  - Positive: rating >= 7/10
  - Negative: rating <= 4/10

## Pipeline Summary

The main data pipeline is implemented in `src/pipeline-utils.py`:

- Extracts text and ratings from the dataset folders (`pos` and `neg` for both train and test sets), assuming the IMDB dataset is already extracted and available in the expected directory structure.
- Assigns sentiment labels based on folder (positive/negative).
- Aggregates all data into a single CSV file (`aclImdb_reviews.csv`) with columns: `label`, `text`, and `rating`.
- Logging is used to track extraction and saving progress.

A Jupyter notebook (`sentinment-analysis.ipynb`) is provided for further data exploration, preprocessing, and model experimentation. The notebook uses `pandas` for data manipulation.

## Usage

1. **Prepare Data:**
   - Download and extract the IMDB dataset from the [official source](https://ai.stanford.edu/~amaas/data/sentiment/) if you have not already done so.
   - Place the extracted dataset in the `raw_data/aclImdb/` directory so that the folder structure matches what the pipeline expects.
2. **Extract Data:**
   - Run the script in `src/pipeline-utils.py` to extract and aggregate the raw IMDB reviews into a CSV file. The script will process the data in place and does not scan for or download the dataset itself.
3. **Explore and Model:**
   - Use the Jupyter notebook to load the CSV, preprocess the data, and experiment with sentiment classification models.

## Dependencies

- Python 3.10+
- pandas
- numpy

Install dependencies (if not already installed):

```bash
pip install pandas numpy
```

## Notes

- The raw dataset is large; ensure you have sufficient disk space and memory.
- The project structure assumes the IMDB dataset is extracted under `raw_data/aclImdb/` before running any scripts. The pipeline does not scan for or download the dataset automatically.
- For more details on the dataset, see the included dataset README in `raw_data/aclImdb/README`.

## Citation

If you use this dataset or project, please cite:

Maas, Andrew L. et al. (2011). "Learning Word Vectors for Sentiment Analysis." Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142–150.

## Contributors

- [Abbas](https://github.com/abbasatayee)

See the [GitHub contributors graph](../../graphs/contributors) for everyone who has contributed to this project.
