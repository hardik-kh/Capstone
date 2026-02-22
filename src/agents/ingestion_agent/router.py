from fastapi import APIRouter, UploadFile, File
from typing import List

from src.agents.ingestion_agent.ingestion_agent import ingest_files

router = APIRouter()


@router.post("/")
async def ingest(files: List[UploadFile] = File(...)):
    """
    Upload CSV or Excel files for ingestion.
    """
    return await ingest_files(files)
