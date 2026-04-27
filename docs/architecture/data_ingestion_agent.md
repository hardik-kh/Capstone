# Data Ingestion Agent

## Purpose

The ingestion agent is the backend entry point for uploaded datasets. It validates uploads, parses CSV/Excel files, profiles tables, persists bronze artifacts, and orchestrates downstream agents (merge, statistics, EDA, predictive, reporting).

Primary implementation:
- `src/agents/ingestion_agent/ingestion_agent.py`
- `src/agents/ingestion_agent/router.py`

## Supported Inputs

- CSV: `.csv`
- Excel: `.xlsx`, `.xls`

Validation is extension-based and returns structured ingestion errors on unsupported formats.

## Pipeline Behavior

### 1. Upload staging

- Files are streamed to temporary disk files in chunks.
- This avoids loading large uploads fully into memory.

### 2. CSV ingestion

- Detects encoding and delimiter.
- Uses full read for normal files.
- Uses sample read for large files (threshold-based).
- Persists bronze artifacts:
  - Normal CSVs: parquet preferred, CSV fallback.
  - Large CSVs: copied directly to bronze CSV.

### 3. Excel ingestion

- Loads workbook sheets and processes each sheet as a separate logical table.
- Each sheet is saved to bronze as CSV.

### 4. Validation and profiling

Row validation reports:
- total rows
- valid/invalid rows
- duplicate rows
- fully-null columns

Cleaning/profile behavior:
- normalize column names to snake_case
- coerce date-like columns to datetime where possible
- fill missing values (context-aware)
- remove exact duplicates
- detect outliers via IQR and report them

Important:
- Outliers are reported only; values are not clipped.
- Negative values are allowed and not treated as invalid by default.

### 5. Merge stage

- Triggered for CSV artifacts.
- For one CSV: saves processed copy (no merge).
- For two or more CSVs: attempts all unique file pairs.
- Merge key inference uses Azure OpenAI with deterministic fallback heuristics.
- DuckDB performs stream-based merge execution and fan-out guard checks.
- Results are returned as a list in `merge_results` (not `merge_result`).

## Output Shape

Main response sections:
- `tables`
- `errors`
- `merge_results`
- `statistical_results`
- `eda_results`
- `predictive_results`
- `reporting_results`
- `processing_log`

## API Endpoints

- `POST /ingest/`  
  Synchronous full pipeline.

- `POST /ingest/start`  
  Starts background ingestion job, returns `job_id`.

- `GET /ingest/status/{job_id}`  
  Returns incremental status and partial/final payload.

## Notes

- Merge inference uses Azure OpenAI configuration from `src/core/config.py`.
- The ingestion router stores job state in-memory for polling.
- CSV merge processing is pairwise for N files (combinational), not restricted to exactly two files.
