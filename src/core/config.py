# Central configuration for ingestion behavior

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_OUTPUT_DIR = DATA_DIR / "bronze"
PROCESSED_OUTPUT_DIR = DATA_DIR / "pro"
EDA_OUTPUT_DIR = DATA_DIR / "eda"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}  # Allowed file types
CSV_DELIMITERS = [",", "\t", ";", "|"]  # Delimiters to try for CSV
ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]  # Encodings to try
MAX_PREVIEW_ROWS = 10  # Number of rows to include in preview
MAX_CATEGORY_VALUES = 10  # Top N categorical values to show
MERGED_OUTPUT_DIR = str(PROCESSED_OUTPUT_DIR)  # Server-side location for processed and merged CSV output
MERGE_SAMPLE_ROWS = 5  # Number of sample rows per dataset shared with the model
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # Stream upload chunks to disk in 1 MB blocks
MERGE_CHUNK_ROWS = 100_000  # Number of CSV rows to process per merge chunk
MERGE_IN_MEMORY_BUILD_MAX_BYTES = 512 * 1024 * 1024  # Max size of the build-side CSV for chunked merges
LARGE_CSV_THRESHOLD_BYTES = 512 * 1024 * 1024  # Switch to sample-based ingestion for CSV files above 512 MB
LARGE_CSV_SAMPLE_ROWS = 50_000  # Rows to sample when profiling very large CSV files

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")