# Multi-Agent Overview

## Implemented Agents

### 1. Ingestion Agent

- File validation and staging
- CSV/Excel parsing
- Cleaning and profiling
- CSV pairwise merge orchestration (via DuckDB + AI-assisted key inference)

### 2. Statistical Agent

- Schema-aware statistical test selection
- Dynamic execution of supported tests
- Technical interpretation + business-oriented insight text

### 3. EDA Agent

- Chooses high-value visualizations per dataset
- Generates plot artifacts (PNG + base64 payloads)
- Produces structured EDA insights for predictive/reporting stages

### 4. Predictive Agent

- Detects available modeling libraries at runtime
- Selects model/target/features with fallback logic
- Trains, evaluates, and saves predictions

### 5. Reporting Agent

- Computes business KPIs and rankings
- Generates report charts
- Composes executive HTML/PDF reports (with graceful PDF fallback)

## Runtime Flow

`Ingestion -> Merge -> Statistical -> EDA -> Predictive -> Reporting`

The pipeline is orchestrated in code inside `ingestion_agent.py`, with status/progress exposed through ingestion job endpoints.

## Design Principles

- Deterministic fallbacks when AI/model dependencies are unavailable
- Structured outputs per stage for frontend rendering
- Stream-first processing for large datasets
- Clear artifact directories under `data/` for bronze, processed, EDA, and reporting outputs
