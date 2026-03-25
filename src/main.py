from fastapi import FastAPI

# upload to bronze layer 
from agent.data_ingestion.upload_csv import router as csv_router
from agent.data_ingestion.upload_xlsx import router as xlsx_router
from agent.data_ingestion.upload_sql import router as sql_router

# generate validation SQL for all tables in bronze layer
from agent.data_ingestion.generate_validation_sql import router as validation_router
from agent.data_ingestion.run_validation_sql import router as run_validation_router

app = FastAPI()

# Register data ingestion routers to bronze layer
app.include_router(csv_router, prefix="/ingestion", tags=["CSV Ingestion"])
app.include_router(xlsx_router, prefix="/ingestion", tags=["XLSX Ingestion"])
app.include_router(sql_router, prefix="/ingestion", tags=["SQL Ingestion"])

# Register validation SQL generation router
app.include_router(validation_router, prefix="/ingestion", tags=["Validation SQL"])
app.include_router(run_validation_router, prefix="/ingestion", tags=["Validation SQL"])
