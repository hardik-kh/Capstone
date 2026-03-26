# Central configuration for ingestion behavior

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_OUTPUT_DIR = DATA_DIR / "bronze"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}  # Allowed file types
CSV_DELIMITERS = [",", "\t", ";", "|"]  # Delimiters to try for CSV
ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]  # Encodings to try
MAX_PREVIEW_ROWS = 10  # Number of rows to include in preview
MAX_CATEGORY_VALUES = 10  # Top N categorical values to show
MERGED_OUTPUT_DIR = str(DATA_DIR)  # Server-side location for merged CSV output
OLLAMA_MODEL = "llama3"  # Default model used for merge-key selection
MERGE_SAMPLE_ROWS = 5  # Number of sample rows per dataset shared with Ollama
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # Stream upload chunks to disk in 1 MB blocks
MERGE_CHUNK_ROWS = 100_000  # Number of CSV rows to process per merge chunk
MERGE_IN_MEMORY_BUILD_MAX_BYTES = 512 * 1024 * 1024  # Max size of the build-side CSV for chunked merges
