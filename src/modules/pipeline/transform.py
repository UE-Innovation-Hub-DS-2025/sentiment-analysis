from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
import re
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    for resource in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources()
stop_words = set(stopwords.words('english'))

def remove_html(text: str) -> str:
    """Remove HTML tags from text and convert to lowercase."""
    return BeautifulSoup(text, "html.parser").get_text().lower()

def clean_text(text: str) -> str:
    """Remove HTML, punctuation, and extra spaces; convert to lowercase."""
    text = remove_html(text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text: str) -> tuple[str, str]:
    """Tokenize text with and without punctuation, removing stopwords."""
    # With punctuation (after HTML removal)
    text_with_punct = remove_html(text)
    tokens_with_punct = [t.lower() for t in word_tokenize(text_with_punct)]
    tokens_with_punct = [t for t in tokens_with_punct if t not in stop_words]
    # Without punctuation
    text_wo_punct = clean_text(text)
    tokens_wo_punct = [t.lower() for t in word_tokenize(text_wo_punct)]
    tokens_wo_punct = [t for t in tokens_wo_punct if t not in stop_words]
    return ' '.join(tokens_with_punct), ' '.join(tokens_wo_punct)

def process_and_save(
    csv_path: Path,
    parquet_with_punct: Path,
    parquet_wo_punct: Path,
    csv_with_punct: Path,
    csv_wo_punct: Path
):
    """Process CSV, tokenize text, and save to CSV and Parquet in two versions."""
    logger.info(f"Processing data from {csv_path}!")
    df = pd.read_csv(csv_path)
    df[['text_with_punctuation', 'text_without_punctuation']] = df['text'].apply(lambda x: pd.Series(tokenize(x)))
    df = shuffle(df, random_state=42).reset_index(drop=True)
    columns_base = [col for col in ['label', 'rating'] if col in df.columns]
    # With punctuation
    df_with_punct = df[['text_with_punctuation'] + columns_base].rename(columns={'text_with_punctuation': 'text'})
    df_with_punct.to_csv(csv_with_punct, index=False)
    df_with_punct.to_parquet(parquet_with_punct, index=False)
    # Without punctuation
    df_wo_punct = df[['text_without_punctuation'] + columns_base].rename(columns={'text_without_punctuation': 'text'})
    df_wo_punct.to_csv(csv_wo_punct, index=False)
    df_wo_punct.to_parquet(parquet_wo_punct, index=False)
    logger.info("Processed data saved!")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[3]
    data_dir = base_dir / "data"
    process_and_save(
        csv_path=data_dir / "processed" / "aclImdb_reviews_raw_extracted.csv",
        parquet_with_punct=data_dir / "transformed" / "parquet" / "aclImdb_reviews_processed_with_punctuation.parquet",
        parquet_wo_punct=data_dir / "transformed" / "parquet" / "aclImdb_reviews_processed_without_punctuation.parquet",
        csv_with_punct=data_dir / "transformed" / "csv" / "aclImdb_reviews_processed_with_punctuation.csv",
        csv_wo_punct=data_dir / "transformed" / "csv" / "aclImdb_reviews_processed_without_punctuation.csv"
    )
