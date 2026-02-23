from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.agents.ingestion_agent.router import router as ingestion_router

app = FastAPI(
    title="Autonomous Analytics",
    version="0.1.0"
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
