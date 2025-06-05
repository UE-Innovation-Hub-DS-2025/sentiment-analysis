import pandas as pd
from sklearn.utils import shuffle
import os
import nltk
from nltk.tokenize import word_tokenize
import logging
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download('punkt_tab')
nltk.download('punkt')

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def remove_html(text):
    # Remove HTML tags using regex and convert to lowercase
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).lower()



def clean_text(text):
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase
    text = text.lower()
    # Remove punctuation (including dots, commas, etc.)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    # Use nltk's word_tokenize, convert tokens to lowercase, and remove stop words
    #     
    text_without_punctuation = clean_text(text)
    tokens_without_punctuation = word_tokenize(text_without_punctuation)

    text_with_punctuation = remove_html(text)
    tokens_with_punctuation = word_tokenize(text_with_punctuation)

    tokens_without_stop_words_with_punctuation = [token for token in tokens_with_punctuation if token not in stop_words]
    tokens_without_stop_words_without_punctuation = [token for token in tokens_without_punctuation if token not in stop_words]
    return ' '.join(tokens_without_stop_words_with_punctuation), ' '.join(tokens_without_stop_words_without_punctuation)  


def process_and_save_parquet(
    csv_path='data/processed/aclImdb_reviews_raw_extracted.csv',
    
    parquet_path_output_with_punctuation='data/processed/aclImdb_reviews_processed_with_punctuation.parquet',
    parquet_path_output_without_punctuation='data/processed/aclImdb_reviews_processed_without_punctuation.parquet',
    
    csv_path_output_with_punctuation='data/processed/aclImdb_reviews_processed_with_punctuation.csv',
    csv_path_output_without_punctuation='data/processed/aclImdb_reviews_processed_without_punctuation.csv'
):
    logger.info(f"Processing data!")
    # Read CSV
    df = pd.read_csv(csv_path)
    # Tokenize text and split into two columns
    df[['text_with_punctuation', 'text_without_punctuation']] = df['text'].apply(lambda x: pd.Series(tokenize(x)))
    # Shuffle
    df = shuffle(df, random_state=42).reset_index(drop=True)
    # Only keep text, label, and rating columns for each version
    columns_base = [col for col in ['label', 'rating'] if col in df.columns]
    columns_with_punct = ['text_with_punctuation'] + columns_base
    columns_without_punct = ['text_without_punctuation'] + columns_base

    # Save with punctuation
    df_with_punct = df[columns_with_punct].rename(columns={'text_with_punctuation': 'text'})
    df_with_punct.to_csv(csv_path_output_with_punctuation, index=False)
    df_with_punct.to_parquet(parquet_path_output_with_punctuation, index=False)
    # Save without punctuation
    df_without_punct = df[columns_without_punct].rename(columns={'text_without_punctuation': 'text'})
    df_without_punct.to_csv(csv_path_output_without_punctuation, index=False)
    df_without_punct.to_parquet(parquet_path_output_without_punctuation, index=False)
    logger.info(f"Processed data saved!")


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
        parquet_path_output_with_punctuation=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "transformed" , 'parquet', "aclImdb_reviews_processed_with_punctuation.parquet"
        ),

        parquet_path_output_without_punctuation=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "transformed", 'parquet', "aclImdb_reviews_processed_without_punctuation.parquet"
        ),
        csv_path_output_with_punctuation=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "transformed", 'csv', "aclImdb_reviews_processed_with_punctuation.csv"
        ),
        csv_path_output_without_punctuation=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "transformed",'csv', "aclImdb_reviews_processed_without_punctuation.csv"
        )
    )
