import os
import logging
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_folder(folder_path, label):
    """
    Extract text and rating from all files in a folder and assign a label.
    Returns a list of (label, text, rating) tuples.
    """
    logger.info(f"Extracting text from folder: {folder_path} with label: {label}")
    data = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # Extract rating from filename: id_rating.txt
            try:
                rating = int(file.split("_")[1].split(".")[0])
            except Exception as e:
                logger.warning(f"Could not extract rating from filename {file}: {e}")
                rating = None
            with open(file_path, "r") as f:
                text = f.read()
                data.append((label, text, rating))
    return data


# Example usage:
folders_and_labels = [
    (
        os.path.join(
            os.path.dirname(__file__), "..", "raw_data", "aclImdb", "test", "neg"
        ),
        "negative",
    ),
    (
        os.path.join(
            os.path.dirname(__file__), "..", "raw_data", "aclImdb", "test", "pos"
        ),
        "positive",
    ),
    (
        os.path.join(
            os.path.dirname(__file__), "..", "raw_data", "aclImdb", "train", "neg"
        ),
        "negative",
    ),
    (
        os.path.join(
            os.path.dirname(__file__), "..", "raw_data", "aclImdb", "train", "pos"
        ),
        "positive",
    ),
]


all_data = []
for folder, label in folders_and_labels:
    all_data.extend(extract_text_from_folder(folder, label))


logger.info(f"Extracted {len(all_data)} rows of data")
# Save to CSV
# Ensure the data directory exists
csv_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "aclImdb_reviews_raw_extracted.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "text", "rating"])
    for row in all_data:
        writer.writerow(row)

logger.info(f"Saved extracted data to {csv_path}")
