import pandas as pd
import re
from sklearn.utils import shuffle
import os
import nltk
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)
nltk.download('punkt_tab')
nltk.download('punkt')

def remove_html(text):
    # Remove HTML tags using regex and convert to lowercase
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).lower()


def tokenize(text):
    # Use nltk's word_tokenize and convert tokens to lowercase
    return [token.lower() for token in word_tokenize(text)]


def process_and_save_parquet(
    csv_path='data/processed/aclImdb_reviews_raw_extracted.csv',
    parquet_path='data/processed/aclImdb_reviews_processed.parquet',
):
    logger.info(f"Processing data from {csv_path} and saving to {parquet_path}")
    # Read CSV
    df = pd.read_csv(csv_path)
    # Remove HTML from text
    df['text'] = df['text'].astype(str).apply(remove_html)
    # Tokenize text
    df['tokens'] = df['text'].apply(tokenize)
    # Shuffle
    df = shuffle(df, random_state=42).reset_index(drop=True)
    # Save to Parquet
    df.to_parquet(parquet_path, index=False)
    print(f"Processed data saved to {parquet_path}")


if __name__ == "__main__":
    # Ensure nltk punkt is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    process_and_save_parquet(
        csv_path=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "processed", "aclImdb_reviews_raw_extracted.csv"
        ),
        parquet_path=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "transformed", "aclImdb_reviews_processed.parquet"
        ),
    )
