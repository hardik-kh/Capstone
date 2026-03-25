from datetime import datetime, timezone

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from agent.data_ingestion.common import IngestionError
from agent.data_ingestion.generate_validation_sql import (
    _build_generated_payload,
    _execute_silver_sql,
    _load_generated_sql_file,
    _write_generated_sql_file,
)

router = APIRouter()


@router.post("/RunValidationSQL")
def run_validation_sql(regenerate: bool = Query(default=False)):
    try:
        generated_payload = _build_generated_payload() if regenerate else _load_generated_sql_file()
        execution_results = _execute_silver_sql(generated_payload["tables"])
        generated_payload["executed"] = True
        generated_payload["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
        generated_payload["execution_results"] = execution_results
        output_file = _write_generated_sql_file(generated_payload)

        return JSONResponse(
            status_code=200,
            content={
                **generated_payload,
                "generated_sql_file": output_file,
            },
        )

    except IngestionError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected validation execution error: {exc}"},
        )
