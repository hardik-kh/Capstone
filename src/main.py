from fastapi import FastAPI
from src.agents.ingestion_agent.router import router as ingestion_router

app = FastAPI(
    title="Autonomous Analytics – Data Ingestion Agent",
    version="0.1.0"
)

app.include_router(ingestion_router, prefix="/ingest", tags=["Data Ingestion"])
