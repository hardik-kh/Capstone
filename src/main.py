import sys
from pathlib import Path

# Support both launch modes:
# 1) project root: `uvicorn src.main:app --reload`
# 2) src directory: `uvicorn main:app --reload`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from src.agents.ingestion_agent.router import router as ingestion_router
from src.agents.reporting_agent.router import router as reporting_router


app = FastAPI(
    title="Autonomous Analytics",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Home route
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Autonomous Analytics</title>
        </head>
        <body style="font-family: Arial; padding: 40px;">
            <h1>Autonomous Analytics</h1>
            <p>Welcome to the Autonomous Analytics API.</p>
            <p>Visit <a href="/docs">/docs</a> to use the ingestion agent.</p>
        </body>
    </html>
    """

# Existing ingestion routes
app.include_router(ingestion_router, prefix="/ingest", tags=["Data Ingestion"])
app.include_router(reporting_router, prefix="/reporting", tags=["Reporting"])
