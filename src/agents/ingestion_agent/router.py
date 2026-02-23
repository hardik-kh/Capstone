# FastAPI router exposing the ingestion endpoint

from typing import List

from fastapi import APIRouter, UploadFile, File

from src.agents.ingestion_agent.ingestion_agent import ingest_files

router = APIRouter()


@router.post("/", summary="Ingest CSV/Excel files", tags=["Data Ingestion"])
async def ingest(files: List[UploadFile] = File(...)):
    """Uploads one or more CSV/Excel files and returns profiling metadata."""
    return await ingest_files(files)
