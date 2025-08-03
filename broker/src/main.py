#!/usr/bin/env python3
"""
FastAPI server for the Compliance Broker.

This module provides a REST API for the compliance broker, allowing clients
to submit compliance requests and receive structured responses from multiple SLMs.
"""

import asyncio
import logging
import uvicorn
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from broker import ComplianceBroker, TaskRequest, BrokerResponse


# Pydantic models for API
class ComplianceRequest(BaseModel):
    """Request model for compliance processing."""
    prompt: str = Field(..., description="Compliance request prompt")
    frameworks: Optional[List[str]] = Field(None, description="Specific frameworks to use")
    max_response_length: int = Field(500, description="Maximum response length")
    temperature: float = Field(0.7, description="Generation temperature")
    include_evidence: bool = Field(True, description="Include evidence requirements")
    include_documents: bool = Field(True, description="Include required documents")


class ComplianceResponse(BaseModel):
    """Response model for compliance processing."""
    original_prompt: str
    combined_response: str
    total_processing_time: float
    frameworks_used: List[str]
    overall_confidence: float
    responses: List[Dict[str, Any]]


class ModelStatus(BaseModel):
    """Model status information."""
    framework: str
    description: str
    is_loaded: bool
    model_path: str
    keywords: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    total_models: int
    uptime: float


# Initialize FastAPI app
app = FastAPI(
    title="Regulatory Compliance SLM Broker",
    description="API for orchestrating multiple small language models for regulatory compliance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global broker instance
broker: Optional[ComplianceBroker] = None
start_time = None


@app.on_event("startup")
async def startup_event():
    """Initialize the broker on startup."""
    global broker, start_time
    import time
    
    start_time = time.time()
    
    # Initialize broker
    models_dir = Path("models/deployed")
    broker = ComplianceBroker(str(models_dir))
    
    # Auto-register models if they exist
    await auto_register_models()
    
    logging.info("Compliance Broker API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global broker
    if broker:
        broker.cleanup()
    logging.info("Compliance Broker API shutdown")


async def auto_register_models():
    """Automatically register models found in the models directory."""
    global broker
    
    models_dir = Path("models/deployed")
    if not models_dir.exists():
        logging.warning(f"Models directory does not exist: {models_dir}")
        return
    
    # Look for model directories
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            framework = model_dir.name.lower()
            
            # Check if it's a valid framework
            if framework in ["sox", "gaap", "fda", "hipaa", "gdpr"]:
                model_path = model_dir / "pytorch_model.bin"
                tokenizer_path = model_dir / "tokenizer.json"
                
                if model_path.exists() and tokenizer_path.exists():
                    success = broker.register_model(framework, str(model_dir))
                    if success:
                        logging.info(f"Auto-registered model: {framework}")
                    else:
                        logging.warning(f"Failed to auto-register model: {framework}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Regulatory Compliance SLM Broker API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global broker, start_time
    import time
    
    if not broker:
        raise HTTPException(status_code=503, detail="Broker not initialized")
    
    status = broker.get_model_status()
    models_loaded = sum(1 for model in status.values() if model["is_loaded"])
    total_models = len(status)
    uptime = time.time() - start_time if start_time else 0
    
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        total_models=total_models,
        uptime=uptime
    )


@app.post("/compliance/process", response_model=ComplianceResponse)
async def process_compliance_request(request: ComplianceRequest):
    """Process a compliance request using multiple SLMs."""
    global broker
    
    if not broker:
        raise HTTPException(status_code=503, detail="Broker not initialized")
    
    try:
        # Convert to broker request
        broker_request = TaskRequest(
            prompt=request.prompt,
            frameworks=request.frameworks,
            max_response_length=request.max_response_length,
            temperature=request.temperature,
            include_evidence=request.include_evidence,
            include_documents=request.include_documents
        )
        
        # Process request
        response = await broker.process_request(broker_request)
        
        # Convert to API response
        api_response = ComplianceResponse(
            original_prompt=response.original_prompt,
            combined_response=response.combined_response,
            total_processing_time=response.total_processing_time,
            frameworks_used=response.frameworks_used,
            overall_confidence=response.overall_confidence,
            responses=[asdict(r) for r in response.responses]
        )
        
        return api_response
        
    except Exception as e:
        logging.error(f"Error processing compliance request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/models", response_model=Dict[str, ModelStatus])
async def get_models():
    """Get status of all registered models."""
    global broker
    
    if not broker:
        raise HTTPException(status_code=503, detail="Broker not initialized")
    
    status = broker.get_model_status()
    return {framework: ModelStatus(**model_info) for framework, model_info in status.items()}


@app.post("/models/register")
async def register_model(framework: str, model_path: str, tokenizer_path: Optional[str] = None):
    """Register a new model."""
    global broker
    
    if not broker:
        raise HTTPException(status_code=503, detail="Broker not initialized")
    
    success = broker.register_model(framework, model_path, tokenizer_path)
    
    if success:
        return {"message": f"Model registered for {framework}", "framework": framework}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to register model for {framework}")


@app.post("/models/{framework}/load")
async def load_model(framework: str):
    """Load a specific model into memory."""
    global broker
    
    if not broker:
        raise HTTPException(status_code=503, detail="Broker not initialized")
    
    success = broker.load_model(framework)
    
    if success:
        return {"message": f"Model loaded for {framework}", "framework": framework}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model for {framework}")


@app.post("/models/{framework}/unload")
async def unload_model(framework: str):
    """Unload a specific model from memory."""
    global broker
    
    if not broker:
        raise HTTPException(status_code=503, detail="Broker not initialized")
    
    success = broker.unload_model(framework)
    
    if success:
        return {"message": f"Model unloaded for {framework}", "framework": framework}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to unload model for {framework}")


@app.get("/frameworks")
async def get_supported_frameworks():
    """Get list of supported regulatory frameworks."""
    return {
        "sox": "Sarbanes-Oxley Act compliance for financial reporting and internal controls",
        "gaap": "Generally Accepted Accounting Principles for financial accounting",
        "fda": "FDA regulations for drug approval and clinical trials",
        "hipaa": "HIPAA compliance for healthcare data privacy and security",
        "gdpr": "GDPR compliance for data protection and privacy"
    }


@app.post("/test")
async def test_compliance():
    """Test endpoint with a sample compliance request."""
    test_request = ComplianceRequest(
        prompt="Generate SOX-compliant internal control documentation for our ERP system",
        frameworks=["sox"],
        max_response_length=300,
        temperature=0.7
    )
    
    return await process_compliance_request(test_request)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 