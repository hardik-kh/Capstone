# Autonomous Analytics Platform

A multi-agent analytics system that ingests CSV/Excel files and produces:
- data profiling and validation
- merge outputs
- statistical analysis
- EDA visuals
- predictive modeling
- executive reporting

## Architecture

The backend is a FastAPI service with specialized agents under `src/agents/`:
- `ingestion_agent`: upload handling, profiling, merge orchestration
- `statistical_agent`: hypothesis/statistical tests
- `eda_agent`: automatic exploratory plots + insights
- `predictive_agent`: model selection and forecasting/prediction
- `reporting_agent`: executive report generation (HTML/PDF routes)

## Project Structure

```text
src/
  main.py
  core/
  agents/
    ingestion_agent/
    statistical_agent/
    eda_agent/
    predictive_agent/
    reporting_agent/
analytiq.html
requirements.txt
data/
```

## Prerequisites

- Python 3.9+
- pip/virtualenv recommended

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in project root (optional for Azure/OpenAI-powered decisions):

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

If these are missing, some AI-assisted parts fall back to deterministic/default behavior.

## Run Backend

```bash
uvicorn src.main:app --reload
```

Service URLs:
- API home: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

## Frontend

Main UI file:
- `analytiq.html`

Open it in a browser (or serve it via a static server). It calls backend endpoints at `http://127.0.0.1:8000`.

## API Endpoints

### Ingestion
- `POST /ingest/`  
  Legacy synchronous endpoint (returns final payload after full run)

- `POST /ingest/start`  
  Starts background ingestion job and returns `job_id`

- `GET /ingest/status/{job_id}`  
  Poll incremental progress and partial results

### Reporting
- `GET /reporting/{dataset_name}/html`  
  Opens generated report in browser

- `GET /reporting/{dataset_name}/pdf`  
  Returns generated PDF if available, otherwise fallback behavior

- `GET /reporting/{dataset_name}/pdf/download`  
  Strict PDF attachment download (requires actual PDF to exist)

- `GET /reporting/`  
  Lists available reports

## Output Directories

Configured in `src/core/config.py`:
- `data/bronze` for ingestion artifacts
- `data/pro` for processed/merged/prediction outputs
- `data/eda` for EDA artifacts
- `data/reporting` for report artifacts

## Notes

- Large CSV handling is enabled (sampling/streaming thresholds in `config.py`).
- CORS is enabled for localhost origins.
- If backend PDF generation dependencies are missing, use report open/print flow as fallback.

## Troubleshooting

1. Backend not reachable
- Confirm `uvicorn` is running on `127.0.0.1:8000`.

2. Upload works but AI-powered choices are weak
- Check Azure/OpenAI env vars.

3. PDF route returns missing/not found
- Verify reporting step completed and PDF dependencies are installed.

## License

Add your preferred license here.
