"""
FastAPI Main Application
Interactive Causal Policy Optimization Lab
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import uuid

from app.routers import data, causal, intervention, optimization, reports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Interactive Causal Policy Optimization Lab",
    version="1.0.0",
    description="""
    A comprehensive platform for causal policy optimization featuring:
    
    - **Data Validation**: Upload and validate tabular datasets
    - **Causal Discovery**: Learn DAG structures using hybrid methods
    - **Intervention Analysis**: Compute interventional distributions P(Y|do(X=x))
    - **Counterfactual Reasoning**: Generate counterfactual samples
    - **Policy Optimization**: Maximize expected utility under constraints
    """,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Session middleware
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    # Get or create session ID
    session_id = request.headers.get("X-Session-ID")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    request.state.session_id = session_id
    
    response = await call_next(request)
    response.headers["X-Session-ID"] = session_id
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Include routers
app.include_router(data.router, prefix="/api/data", tags=["Data Management"])
app.include_router(causal.router, prefix="/api/causal", tags=["Causal Analysis"])
app.include_router(intervention.router, prefix="/api/intervention", tags=["Intervention & Counterfactual"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Policy Optimization"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports & Export"])


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "app_name": "Interactive Causal Policy Optimization Lab",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Interactive Causal Policy Optimization Lab",
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )