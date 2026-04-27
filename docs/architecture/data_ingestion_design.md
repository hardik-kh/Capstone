# Data Ingestion Design

## Design Goals

- Accept mixed CSV/Excel uploads safely.
- Support large CSV workloads without exhausting memory.
- Produce consistent profiling metadata for downstream agents.
- Preserve staged/raw artifacts for reproducibility.
- Continue processing when individual files fail.

## Core Components

- `router.py`: FastAPI endpoints and async job polling.
- `ingestion_agent.py`: end-to-end orchestration.
- `csv_handler.py`: CSV detection/read and bronze persistence.
- `excel_handler.py`: workbook/sheet loading and bronze persistence.
- `validators.py`: file-level and row-level validation metrics.
- `profiler.py`: cleaning + profiling outputs.
- `merge_service.py`: pairwise CSV merge inference/execution via DuckDB.

## Key Design Choices

### Upload streaming

Uploads are written to temp files in chunks to avoid loading entire payloads into memory.

### Large CSV strategy

For files above threshold:
- profile from sampled rows
- copy full file directly to bronze CSV
- still allow merge on original staged file

### Cleaning/profile contract

Profiling returns schema/summary/preview and quality metrics. Outliers are detected and reported, not modified.

### Merge strategy

- Works across all unique CSV pairs for N uploaded CSV files.
- Uses Azure OpenAI merge-key inference with heuristic fallback.
- Uses DuckDB streaming joins and fan-out guardrails.

### Resilience

- Structured errors are collected in `errors`.
- Pipeline continues across files and stages where possible.
- Progress snapshots are emitted for async job polling.
