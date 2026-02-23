# Central configuration for ingestion behavior

MAX_FILE_SIZE_MB = 50  # Maximum upload size per file
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}  # Allowed file types
CSV_DELIMITERS = [",", "\t", ";", "|"]  # Delimiters to try for CSV
ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]  # Encodings to try
MAX_PREVIEW_ROWS = 10  # Number of rows to include in preview
MAX_CATEGORY_VALUES = 10  # Top N categorical values to show
