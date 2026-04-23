# FastAPI router exposing ingestion endpoints

import asyncio
import os
import tempfile
import time
import copy
from uuid import uuid4
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from agents.ingestion_agent.ingestion_agent import ingest_files

router = APIRouter()

_JOBS_LOCK = asyncio.Lock()
_INGESTION_JOBS: dict[str, dict] = {}


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


async def _upsert_job(job_id: str, **updates) -> None:
    async with _JOBS_LOCK:
        if job_id not in _INGESTION_JOBS:
            _INGESTION_JOBS[job_id] = {"job_id": job_id}
        _INGESTION_JOBS[job_id].update(updates)
        _INGESTION_JOBS[job_id]["updated_at"] = _utc_now()


async def _get_job(job_id: str) -> Optional[dict]:
    async with _JOBS_LOCK:
        job = _INGESTION_JOBS.get(job_id)
        return copy.deepcopy(job) if job else None


async def _persist_uploads(files: List[UploadFile]) -> list[dict]:
    persisted: list[dict] = []
    for file in files:
        suffix = os.path.splitext(file.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
            persisted.append({"filename": file.filename, "path": tmp.name})
    return persisted


async def _run_ingestion_job(job_id: str, persisted_uploads: list[dict]) -> None:
    upload_handles: list[UploadFile] = []
    started = time.time()
    try:
        await _upsert_job(job_id, status="running")
        for item in persisted_uploads:
            upload_handles.append(UploadFile(filename=item["filename"], file=open(item["path"], "rb")))

        async def _progress_callback(snapshot: dict) -> None:
            await _upsert_job(job_id, status="running", data=snapshot)

        final_payload = await ingest_files(upload_handles, progress_callback=_progress_callback)
        await _upsert_job(
            job_id,
            status="completed",
            data=final_payload,
            duration_seconds=round(time.time() - started, 4),
        )
    except Exception as e:
        await _upsert_job(
            job_id,
            status="failed",
            error=str(e),
            duration_seconds=round(time.time() - started, 4),
        )
    finally:
        for handle in upload_handles:
            try:
                await handle.close()
            except Exception:
                pass
        for item in persisted_uploads:
            try:
                if os.path.exists(item["path"]):
                    os.remove(item["path"])
            except Exception:
                pass


@router.post("/", summary="Ingest files and merge CSV uploads", tags=["Data Ingestion"])
async def ingest(files: List[UploadFile] = File(...)):
    """Legacy synchronous endpoint: returns full payload after all agents finish."""
    return await ingest_files(files)


@router.post("/start", summary="Start ingestion job", tags=["Data Ingestion"])
async def ingest_start(files: List[UploadFile] = File(...)):
    """Starts ingestion in background and returns a job_id for polling status."""
    job_id = uuid4().hex
    persisted_uploads = await _persist_uploads(files)
    await _upsert_job(
        job_id,
        status="queued",
        started_at=_utc_now(),
        files=[f.filename for f in files],
        data={
            "tables": [],
            "errors": [],
            "merge_results": [],
            "statistical_results": [],
            "eda_results": [],
            "predictive_results": [],
            "reporting_results": [],
            "processing_log": {
                "started_at": _utc_now(),
                "files_received": [f.filename for f in files],
                "events": [],
            },
        },
    )
    asyncio.create_task(_run_ingestion_job(job_id, persisted_uploads))
    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}", summary="Get ingestion job status", tags=["Data Ingestion"])
async def ingest_status(job_id: str):
    """Poll this endpoint to get incremental partial results."""
    job = await _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    return JSONResponse(
        content=job,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
