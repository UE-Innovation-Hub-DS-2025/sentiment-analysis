import sys
from src.modules.pipeline.transform import process_and_save_parquet

if __name__ == "__main__":
    # Optionally accept custom input/output paths from command line
    if len(sys.argv) == 3:
        csv_path = sys.argv[1]
        parquet_path = sys.argv[2]
        process_and_save_parquet(csv_path=csv_path, parquet_path=parquet_path)
    else:
        process_and_save_parquet() 