# FastAPI router exposing the ingestion endpoint

from typing import List

from fastapi import APIRouter, UploadFile, File

from agents.ingestion_agent.ingestion_agent import ingest_files

router = APIRouter()


@router.post("/", summary="Ingest files and merge exactly two CSV uploads", tags=["Data Ingestion"])
async def ingest(files: List[UploadFile] = File(...)):
    """Uploads files, profiles them, and merges exactly two CSV uploads when possible."""
    return await ingest_files(files)
