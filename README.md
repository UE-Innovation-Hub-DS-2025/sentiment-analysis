# Sentiment Analysis on IMDB Movie Reviews

## Overview

This project performs sentiment analysis on the IMDB Large Movie Review Dataset. The goal is to classify movie reviews as either positive or negative using machine learning techniques. The project includes data extraction, preprocessing, and preparation for model training and evaluation.

## Dataset

The dataset used is the [Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 labeled reviews (25,000 for training and 25,000 for testing) and 50,000 unlabeled reviews for unsupervised learning. Reviews are split into positive and negative categories, and each review is stored as a text file with its sentiment label and rating encoded in the filename.

- **Source:** See `raw_data/aclImdb/README` for full dataset details and citation.
- **Labels:**
  - Positive: rating >= 7/10
  - Negative: rating <= 4/10

## Pipeline Summary

The main data pipeline is implemented in `src/pipeline-utils.py`:

- Extracts text and ratings from the dataset folders (`pos` and `neg` for both train and test sets).
- Assigns sentiment labels based on folder (positive/negative).
- Aggregates all data into a single CSV file (`aclImdb_reviews.csv`) with columns: `label`, `text`, and `rating`.
- Logging is used to track extraction and saving progress.

A Jupyter notebook (`sentinment-analysis.ipynb`) is provided for further data exploration, preprocessing, and model experimentation. The notebook uses `pandas` for data manipulation.

## Usage

1. **Extract Data:**
   - Run the script in `src/pipeline-utils.py` to extract and aggregate the raw IMDB reviews into a CSV file.
2. **Explore and Model:**
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
- The project structure assumes the IMDB dataset is extracted under `raw_data/aclImdb/`.
- For more details on the dataset, see the included dataset README in `raw_data/aclImdb/README`.

## Citation

If you use this dataset or project, please cite:

Maas, Andrew L. et al. (2011). "Learning Word Vectors for Sentiment Analysis." Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142–150.

## Contributors

- [Abbas](https://github.com/abbasatayee)

See the [GitHub contributors graph](../../graphs/contributors) for everyone who has contributed to this project.
