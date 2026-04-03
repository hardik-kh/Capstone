# Data Ingestion Agent

## Purpose

The Data Ingestion Agent is the system entry point for uploaded datasets. It accepts CSV and Excel files, validates them, stages them to disk, profiles their contents, stores raw copies for downstream use, and optionally merges exactly two CSV files into a processed output.

The implementation lives primarily in [src/agents/ingestion_agent/ingestion_agent.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/ingestion_agent.py) and is exposed through the FastAPI route in [src/agents/ingestion_agent/router.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/router.py).

## What The Agent Does

### 1. Accepts uploaded files

The API accepts multiple uploaded files in a single request. Supported file types are:

- `.csv`
- `.xlsx`
- `.xls`

Unsupported file types are rejected during validation.

### 2. Streams uploads to temporary storage

Each upload is written to a temporary file in chunks instead of being held fully in memory. This reduces memory pressure during upload and makes it possible to handle large files safely.

### 3. Validates file type

Before any parsing happens, the agent checks the uploaded file extension against the allowed formats. Invalid formats return a structured ingestion error.

### 4. Processes CSV files

For CSV uploads, the agent:

- Detects encoding
- Detects delimiter
- Loads the file into pandas for normal-sized files
- Switches to sample-based loading for large CSV files
- Saves a bronze copy for raw persistence
- Runs row-level validation
- Cleans and profiles the dataset
- Records processing steps and timing metadata

#### Large CSV behavior

Large CSVs are handled differently to avoid exhausting memory. When the temporary file size crosses the configured threshold:

- The agent reads only a bounded sample of rows for profiling and validation
- The full CSV is copied directly into bronze storage without loading the entire file into pandas
- The upload can still participate in merge processing later through the DuckDB-based merge path

This is what allows the system to support files such as a `train.csv` around 5 GB in size.

### 5. Processes Excel files

For Excel uploads, the agent:

- Reads all sheets from the workbook
- Cleans and profiles each sheet independently
- Returns each sheet as a separate logical table in the response

Excel files are profiled sheet by sheet, but they are not part of the CSV merge flow.

### 6. Profiles datasets

The profiling step performs lightweight cleaning and descriptive analysis. It currently:

- Normalizes column names to snake_case
- Attempts date coercion for date-like columns
- Fills missing values
- Removes duplicate rows
- Clips numeric outliers using the IQR rule

The returned profiling metadata includes:

- Shape before and after cleaning
- Missing value counts
- Data types
- Numeric summary statistics
- Top categorical values
- Preview rows

### 7. Validates rows

The agent computes simple row-level validation metrics, including:

- Total rows
- Valid rows
- Invalid rows
- Validation coverage percentage

Current validation rules are intentionally simple:

- Rows cannot be completely null
- Numeric values cannot be negative

### 8. Saves bronze copies

CSV files are persisted to the bronze layer for raw storage:

- Normal-sized CSVs are saved as Parquet when possible, with CSV fallback
- Large CSVs are copied directly as CSV to avoid expensive in-memory conversion

This gives downstream agents access to a stored raw version of the upload.

### 9. Merges exactly two CSV files when possible

If and only if exactly two CSV uploads succeed, the ingestion agent invokes the merge service.

The merge service:

- Loads bounded samples from each CSV for merge-key inference
- Uses Ollama to infer the most appropriate shared join key
- Supports single-column keys and composite keys
- Falls back to deterministic heuristics if model output is missing or invalid
- Uses DuckDB to execute the merge directly from the CSV files
- Writes the merged output to the processed data directory
- Also writes normalized processed copies of both source CSVs

If no shared columns are found, the agent skips the merge and still returns the two processed source copies.

## Output Structure

The ingestion response currently returns four top-level sections:

- `tables`
- `errors`
- `merge_result`
- `processing_log`

### `tables`

Contains one entry per ingested table:

- One entry per CSV file
- One entry per Excel sheet

Each table entry includes:

- Table name
- Source type
- Optional sheet name
- Profiling results
- Ingestion metadata

### `errors`

Contains structured ingestion or merge errors. The agent continues processing other files when possible instead of failing the entire request immediately.

### `merge_result`

Returned when exactly two CSV files are available for merge processing. It includes:

- Merged output path
- Merged output filename
- Row and column counts for the merged output
- `copied_tables` for the normalized processed source files
- Detailed merge summary

### `processing_log`

Contains end-to-end execution details, including:

- Start and completion timestamps
- Files received
- Per-file processing steps
- Merge step details
- Status markers for success, failure, or skipped work

This log is useful for debugging, auditability, and UI progress reporting.

## Merge Strategy

The merge system is designed to be safe for larger datasets and more reliable than a naive shared-column join.

### Merge-key inference

The merge service analyzes:

- Shared column names
- Column uniqueness
- Null rates
- Sample values
- Overlap between candidate keys

It then asks Ollama to choose:

- The best join type
- One shared column, or
- Two shared columns as a composite key when that is safer

If Ollama returns unusable output, the service uses a heuristic fallback that prioritizes business identifiers such as `store_nbr`, `item_nbr`, and related fields.

### Merge execution

Merge execution is handled by DuckDB, not pandas. This matters because DuckDB can operate directly on CSV files and avoids reading entire multi-GB files into Python memory before joining.

The merged output:

- Is written as a CSV file
- Uses normalized column names
- Prefixes left and right columns with their source dataset names

## Storage Layers

The agent currently uses two output areas under `data/`:

- `data/bronze`
- `data/pro`

### `data/bronze`

Raw persisted copies of ingested CSV data.

### `data/pro`

Processed copies and merged outputs generated by the merge service.

## Example CSVs

Below are small example shapes based on the Favorita-style datasets the platform is designed around.

### Example `train.csv`

```csv
id,date,store_nbr,item_nbr,unit_sales,onpromotion
1,2013-01-01,1,103665,7,False
2,2013-01-01,1,105574,1,False
3,2013-01-01,2,103665,2,True
4,2013-01-02,1,103665,5,False
```

Typical columns:

- `date`: sales date
- `store_nbr`: store identifier
- `item_nbr`: item identifier
- `unit_sales`: units sold
- `onpromotion`: promotion flag

### Example `transactions.csv`

```csv
date,store_nbr,transactions
2013-01-01,1,2111
2013-01-01,2,2358
2013-01-02,1,1897
2013-01-02,2,2214
```

Typical columns:

- `date`: transaction date
- `store_nbr`: store identifier
- `transactions`: transaction count for that store and date

### Expected merge behavior for these files

For `train.csv` and `transactions.csv`, the most reliable merge is usually a composite key using:

- `store_nbr`
- `date`

That is because `store_nbr` alone is often too broad, and `date` alone is almost never enough. Together they identify store-level daily transaction context much more safely.

## Running The Merge Service

The repository includes a helper script at [run_merge_service.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/run_merge_service.py) for merging two CSV files from the `data/` directory.

### Prerequisites

- Put both CSV files inside `data/`
- Make sure Python dependencies are installed
- Make sure Ollama is running in the background if you want model-based merge-key inference

### Important note about Ollama

Ollama should be running in the background before you run the merge service. If Ollama is not running, merge-key inference through the model will fail and the service may fall back to heuristics or raise an error depending on the environment.

Example way to start Ollama on your machine:

```bash
ollama serve
```

Keep that process running in a separate terminal window or as a background service.

### Example file placement

```text
data/
  train.csv
  transactions.csv
```

### Command

Run the merge service from the project root:

```bash
python3 src/agents/ingestion_agent/run_merge_service.py train.csv transactions.csv
```

### What the command prints

The script prints:

- Left and right file names
- Left and right normalized columns
- Ollama’s merge decision
- Selected merge columns
- Merged output path
- Processed source copy paths
- Full merge log as JSON

### Expected output files

After a successful run, you should typically see outputs in `data/pro/` such as:

- `train__processed.csv`
- `transactions__processed.csv`
- `train__transactions__merged.csv`

## Key Files

- [src/agents/ingestion_agent/router.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/router.py): FastAPI endpoint
- [src/agents/ingestion_agent/ingestion_agent.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/ingestion_agent.py): Main orchestration logic
- [src/agents/ingestion_agent/csv_handler.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/csv_handler.py): CSV loading and bronze persistence
- [src/agents/ingestion_agent/excel_handler.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/excel_handler.py): Excel loading
- [src/agents/ingestion_agent/profiler.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/profiler.py): Cleaning and profiling
- [src/agents/ingestion_agent/validators.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/validators.py): File and row validation
- [src/agents/ingestion_agent/merge_service.py](/Users/dhanraj/workplace/Capstone-FSE570/src/agents/ingestion_agent/merge_service.py): Merge inference and DuckDB-based merge execution
- [src/core/config.py](/Users/dhanraj/workplace/Capstone-FSE570/src/core/config.py): Thresholds and output directories

## Current Limitations

- Only exactly two CSV files are considered for merge in a single ingestion request
- Large CSV profiling is sample-based, so profile statistics for very large files are not computed over the full dataset
- Excel files are profiled but not merged
- Row validation rules are currently basic and may need to become domain-aware over time
- Ollama is used only when shared columns exist and merge inference is needed

## Summary

The Data Ingestion Agent is responsible for getting uploaded data into the platform safely and consistently. It validates files, stages them efficiently, profiles them for downstream intelligence, preserves raw copies, and produces merge-ready processed outputs. Its current design balances transparency, structured outputs, and practical support for large CSV workloads.
