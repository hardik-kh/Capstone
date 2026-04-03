# Data Ingestion Agent  
### Cleaning, Profiling, and Preparing Data for Autonomous Analytics

The Data Ingestion Agent is the foundation of the Autonomous Analytics Platform. It transforms raw uploaded files into **clean, structured, profiled datasets** that downstream agents can consume without manual intervention.

This agent is deterministic, fast, and built using FastAPI + Pandas.

---

# 🎯 Responsibilities

### 1. File Intake
- Accept multiple uploaded files (CSV, XLSX, XLS)
- Validate file extensions and sizes
- Save uploads to temporary storage

### 2. File Parsing
- CSV:
  - Encoding detection (UTF‑8, Latin‑1, CP1252)
  - Delimiter detection (comma, tab, semicolon, pipe)
- Excel:
  - Multi‑sheet loading
  - Each sheet becomes a separate table

---

# 🧹 Data Cleaning

The ingestion agent performs **automatic corrective actions**:

| Issue | Action |
|-------|--------|
| Missing numeric values | Fill with median |
| Missing categorical values | Fill with mode or `"unknown"` |
| Duplicate rows | Remove |
| Outliers | Clip using IQR |
| Date columns | Auto‑convert to datetime |
| Column names | Normalize to snake_case |

These actions ensure downstream agents receive **clean, consistent, analysis‑ready data**.

---

# 📊 Profiling

Each dataset is profiled to generate metadata used by ETL, EDA, Statistical Testing, and Modeling agents.

### Profiling Includes:
- Shape before/after cleaning  
- Missing values before/after  
- Data types  
- Numeric summary statistics  
- Categorical distributions  
- Data preview (first 10 rows)  

---

# ⚠️ Error Handling

All errors follow a structured schema:

```json
{
  "file": "train.csv",
  "error_code": "FILE_READ_ERROR",
  "message": "Failed to read file 'train.csv'.",
  "hint": "Details: Error tokenizing data..."
}