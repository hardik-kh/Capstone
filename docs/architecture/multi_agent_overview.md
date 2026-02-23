# Multi‑Agent System Architecture  
### Autonomous Analytics from Raw Data to Insights

This document describes the multi‑agent architecture powering the Autonomous Analytics Platform. Each agent is specialized, deterministic, and orchestrated using a central workflow engine.

---

# 🧩 Agent Overview

## 1. Data Ingestion Agent
- Validates files  
- Parses CSV/Excel  
- Cleans missing values, outliers, duplicates  
- Profiles datasets  
- Produces structured metadata  

---

## 2. ETL & Data Quality Agent
- Merges Favorita datasets:
  - train.csv  
  - test.csv  
  - stores.csv  
  - items.csv  
  - transactions.csv  
  - oil.csv  
  - holidays_events.csv  
- Handles dataset‑specific quirks:
  - Missing promotions  
  - Negative unit_sales (returns)  
  - Items in test not in train  
  - Transferred holidays  
  - Missing oil prices  
- Creates engineered features:
  - Lag features  
  - Rolling windows  
  - Holiday flags  
  - Perishable multipliers  

---

## 3. Exploratory Analysis Agent
- Correlation analysis  
- Trend detection  
- Seasonality detection  
- Outlier detection  
- Feature importance  

---

## 4. Statistical Testing Agent
- Promotion effect significance  
- Holiday impact tests  
- Store cluster differences  
- Oil price correlation tests  
- Multiple comparison correction  

---

## 5. Predictive Analysis Agent
- Forecasts unit sales  
- Handles unseen items  
- Produces confidence intervals  
- Supports:
  - LightGBM  
  - XGBoost  
  - Prophet  

---

## 6. Insight Synthesis Agent
- Converts analytical outputs into business‑ready insights  
- Summarizes findings  
- Ranks insights by impact  
- Generates recommendations  

---

## 7. Orchestrator Agent (LangGraph)
Coordinates the entire workflow:

Ingestion
↓
ETL & Data Quality
↓
Exploratory Analysis
↓
Statistical Testing
↓
Predictive Modeling
↓
Insight Synthesis

--- 

# Responsibilities:
- State management  
- Error propagation  
- Conditional branching  
- Retry logic  
- Workflow completion  

---

# 🧠 Design Principles

- Deterministic processing for ingestion, ETL, statistics, modeling  
- LLM‑only for insight synthesis  
- Modular agents with clear boundaries  
- Reproducible pipelines  
- Dataset‑aware intelligence  
- Scalable architecture  

---

# 🎯 Summary

This multi‑agent system enables **fully autonomous analytics**, transforming raw data into insights without requiring data science expertise. Each agent is specialized, testable, and orchestrated through a robust workflow engine.
