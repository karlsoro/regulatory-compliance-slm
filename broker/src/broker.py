"""
Broker for Orchestrating Multiple Regulatory Compliance SLMs

This module provides functionality to coordinate multiple small language models
trained on different regulatory frameworks to handle complex compliance requests.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a deployed SLM."""
    framework: str
    model_path: str
    tokenizer_path: str
    description: str
    keywords: List[str]
    is_loaded: bool = False
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None


@dataclass
class TaskRequest:
    """Request for compliance task processing."""
    prompt: str
    frameworks: Optional[List[str]] = None
    max_response_length: int = 500
    temperature: float = 0.7
    include_evidence: bool = True
    include_documents: bool = True


@dataclass
class TaskResponse:
    """Response from a compliance task."""
    framework: str
    response: str
    confidence: float
    required_documents: List[str]
    evidence_requirements: List[str]
    processing_time: float
    model_used: str


@dataclass
class BrokerResponse:
    """Combined response from the broker."""
    original_prompt: str
    responses: List[TaskResponse]
    combined_response: str
    total_processing_time: float
    frameworks_used: List[str]
    overall_confidence: float


class ComplianceBroker:
    """
    Broker for orchestrating multiple regulatory compliance SLMs.
    
    This broker coordinates multiple small language models, each trained on
    different regulatory frameworks, to handle complex compliance requests
    that may span multiple regulations.
    """
    
    def __init__(self, models_dir: str = "models/deployed"):
        """
        Initialize the compliance broker.
        
        Args:
            models_dir: Directory containing deployed models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, ModelInfo] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Framework-specific configurations
        self.framework_configs = {
            "sox": {
                "description": "Sarbanes-Oxley Act compliance for financial reporting and internal controls",
                "keywords": ["sox", "sarbanes", "oxley", "financial", "reporting", "internal control", "audit", "section 404", "section 302"],
                "priority": 1
            },
            "gaap": {
                "description": "Generally Accepted Accounting Principles for financial accounting",
                "keywords": ["gaap", "accounting", "financial statement", "revenue", "asset", "liability", "equity"],
                "priority": 2
            },
            "fda": {
                "description": "FDA regulations for drug approval and clinical trials",
                "keywords": ["fda", "clinical trial", "drug approval", "21 cfr", "electronic record", "validation"],
                "priority": 3
            },
            "hipaa": {
                "description": "HIPAA compliance for healthcare data privacy and security",
                "keywords": ["hipaa", "privacy", "security", "patient", "phi", "breach", "authorization"],
                "priority": 4
            },
            "gdpr": {
                "description": "GDPR compliance for data protection and privacy",
                "keywords": ["gdpr", "data protection", "consent", "privacy", "data subject", "controller"],
                "priority": 5
            }
        }
        
        logger.info(f"Compliance Broker initialized")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Models directory: {self.models_dir}")
    
    def register_model(self, framework: str, model_path: str, 
                      tokenizer_path: Optional[str] = None) -> bool:
        """
        Register a model for a specific regulatory framework.
        
        Args:
            framework: Regulatory framework (e.g., "sox", "fda")
            model_path: Path to the model files
            tokenizer_path: Path to the tokenizer files (optional)
            
        Returns:
            True if registration successful, False otherwise
        """
        if framework not in self.framework_configs:
            logger.error(f"Unknown framework: {framework}")
            return False
        
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            logger.error(f"Tokenizer path does not exist: {tokenizer_path}")
            return False
        
        config = self.framework_configs[framework]
        
        model_info = ModelInfo(
            framework=framework,
            model_path=str(model_path),
            tokenizer_path=str(tokenizer_path),
            description=config["description"],
            keywords=config["keywords"]
        )
        
        self.models[framework] = model_info
        logger.info(f"Registered model for {framework}: {model_path}")
        
        return True
    
    def load_model(self, framework: str) -> bool:
        """
        Load a model into memory.
        
        Args:
            framework: Regulatory framework to load
            
        Returns:
            True if loading successful, False otherwise
        """
        if framework not in self.models:
            logger.error(f"Model not registered for framework: {framework}")
            return False
        
        model_info = self.models[framework]
        
        if model_info.is_loaded:
            logger.info(f"Model for {framework} already loaded")
            return True
        
        try:
            logger.info(f"Loading model for {framework}...")
            
            # Load tokenizer
            model_info.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer_path)
            
            # Load model
            model_info.model = AutoModelForCausalLM.from_pretrained(
                model_info.model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            model_info.model.to(self.device)
            model_info.model.eval()
            
            model_info.is_loaded = True
            logger.info(f"Successfully loaded model for {framework}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model for {framework}: {str(e)}")
            return False
    
    def unload_model(self, framework: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            framework: Regulatory framework to unload
            
        Returns:
            True if unloading successful, False otherwise
        """
        if framework not in self.models:
            logger.error(f"Model not registered for framework: {framework}")
            return False
        
        model_info = self.models[framework]
        
        if not model_info.is_loaded:
            logger.info(f"Model for {framework} not loaded")
            return True
        
        try:
            # Clear model and tokenizer
            del model_info.model
            del model_info.tokenizer
            model_info.model = None
            model_info.tokenizer = None
            model_info.is_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"Successfully unloaded model for {framework}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model for {framework}: {str(e)}")
            return False
    
    def detect_relevant_frameworks(self, prompt: str) -> List[str]:
        """
        Detect which regulatory frameworks are relevant to a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            List of relevant framework names
        """
        prompt_lower = prompt.lower()
        relevant_frameworks = []
        
        for framework, config in self.framework_configs.items():
            # Check keyword matches
            keyword_matches = sum(1 for keyword in config["keywords"] 
                                if keyword.lower() in prompt_lower)
            
            # Calculate relevance score
            relevance_score = keyword_matches / len(config["keywords"])
            
            if relevance_score > 0.1:  # Threshold for relevance
                relevant_frameworks.append(framework)
        
        # If no frameworks detected, return all available
        if not relevant_frameworks and self.models:
            relevant_frameworks = list(self.models.keys())
        
        return relevant_frameworks
    
    async def process_request(self, request: TaskRequest) -> BrokerResponse:
        """
        Process a compliance request using multiple SLMs.
        
        Args:
            request: Task request
            
        Returns:
            Combined response from all relevant models
        """
        start_time = time.time()
        
        # Detect relevant frameworks
        if request.frameworks:
            relevant_frameworks = request.frameworks
        else:
            relevant_frameworks = self.detect_relevant_frameworks(request.prompt)
        
        logger.info(f"Processing request for frameworks: {relevant_frameworks}")
        
        # Load models if needed
        for framework in relevant_frameworks:
            if framework in self.models and not self.models[framework].is_loaded:
                self.load_model(framework)
        
        # Process with each relevant model
        responses = []
        for framework in relevant_frameworks:
            if framework in self.models and self.models[framework].is_loaded:
                response = await self._process_with_model(framework, request)
                responses.append(response)
        
        # Combine responses
        combined_response = self._combine_responses(responses, request.prompt)
        
        # Calculate overall metrics
        total_time = time.time() - start_time
        overall_confidence = np.mean([r.confidence for r in responses]) if responses else 0.0
        
        broker_response = BrokerResponse(
            original_prompt=request.prompt,
            responses=responses,
            combined_response=combined_response,
            total_processing_time=total_time,
            frameworks_used=relevant_frameworks,
            overall_confidence=overall_confidence
        )
        
        logger.info(f"Request processed in {total_time:.2f} seconds")
        logger.info(f"Used frameworks: {relevant_frameworks}")
        logger.info(f"Overall confidence: {overall_confidence:.3f}")
        
        return broker_response
    
    async def _process_with_model(self, framework: str, request: TaskRequest) -> TaskResponse:
        """Process request with a specific model."""
        model_info = self.models[framework]
        start_time = time.time()
        
        try:
            # Generate response
            response_text = self._generate_response(
                model_info.model,
                model_info.tokenizer,
                request.prompt,
                max_length=request.max_response_length,
                temperature=request.temperature
            )
            
            # Extract structured information
            required_docs = self._extract_required_documents(response_text, framework)
            evidence_reqs = self._extract_evidence_requirements(response_text, framework)
            
            # Calculate confidence
            confidence = self._calculate_confidence(response_text, framework)
            
            processing_time = time.time() - start_time
            
            return TaskResponse(
                framework=framework,
                response=response_text,
                confidence=confidence,
                required_documents=required_docs,
                evidence_requirements=evidence_reqs,
                processing_time=processing_time,
                model_used=model_info.model_path
            )
            
        except Exception as e:
            logger.error(f"Error processing with {framework} model: {str(e)}")
            processing_time = time.time() - start_time
            
            return TaskResponse(
                framework=framework,
                response=f"Error processing request: {str(e)}",
                confidence=0.0,
                required_documents=[],
                evidence_requirements=[],
                processing_time=processing_time,
                model_used=model_info.model_path
            )
    
    def _generate_response(self, model, tokenizer, prompt: str, 
                          max_length: int = 500, temperature: float = 0.7) -> str:
        """Generate response using a loaded model."""
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _extract_required_documents(self, response: str, framework: str) -> List[str]:
        """Extract required documents from response."""
        documents = []
        
        # Framework-specific document extraction
        doc_keywords = {
            "sox": ["control documentation", "audit report", "financial statement", "internal control assessment"],
            "gaap": ["financial statement", "accounting policy", "revenue recognition", "asset documentation"],
            "fda": ["clinical trial report", "validation protocol", "electronic record", "approval documentation"],
            "hipaa": ["privacy policy", "security assessment", "breach notification", "authorization form"],
            "gdpr": ["data processing agreement", "consent form", "privacy notice", "data protection impact assessment"]
        }
        
        keywords = doc_keywords.get(framework, [])
        
        for keyword in keywords:
            if keyword.lower() in response.lower():
                documents.append(keyword)
        
        return documents
    
    def _extract_evidence_requirements(self, response: str, framework: str) -> List[str]:
        """Extract evidence requirements from response."""
        evidence = []
        
        # Framework-specific evidence extraction
        evidence_keywords = {
            "sox": ["audit trail", "control testing", "management certification", "disclosure controls"],
            "gaap": ["accounting records", "supporting documentation", "reconciliation", "review evidence"],
            "fda": ["validation records", "system logs", "approval documentation", "compliance testing"],
            "hipaa": ["access logs", "security assessments", "training records", "incident reports"],
            "gdpr": ["consent records", "data processing logs", "privacy impact assessments", "breach notifications"]
        }
        
        keywords = evidence_keywords.get(framework, [])
        
        for keyword in keywords:
            if keyword.lower() in response.lower():
                evidence.append(keyword)
        
        return evidence
    
    def _calculate_confidence(self, response: str, framework: str) -> float:
        """Calculate confidence score for a response."""
        # Simple confidence calculation based on response quality
        if not response or len(response.strip()) < 10:
            return 0.0
        
        # Check for framework-specific keywords
        config = self.framework_configs.get(framework, {})
        keywords = config.get("keywords", [])
        
        keyword_matches = sum(1 for keyword in keywords 
                            if keyword.lower() in response.lower())
        keyword_score = keyword_matches / len(keywords) if keywords else 0.0
        
        # Length score (longer responses tend to be more detailed)
        length_score = min(1.0, len(response) / 200)
        
        # Structure score (check for structured elements)
        structure_indicators = ["required", "documentation", "evidence", "procedure", "control"]
        structure_matches = sum(1 for indicator in structure_indicators 
                              if indicator.lower() in response.lower())
        structure_score = structure_matches / len(structure_indicators)
        
        # Combine scores
        confidence = (keyword_score * 0.4 + length_score * 0.3 + structure_score * 0.3)
        
        return min(1.0, confidence)
    
    def _combine_responses(self, responses: List[TaskResponse], original_prompt: str) -> str:
        """Combine responses from multiple models into a coherent response."""
        if not responses:
            return "No relevant models available to process this request."
        
        if len(responses) == 1:
            return responses[0].response
        
        # Create a structured combined response
        combined_parts = []
        combined_parts.append(f"Compliance Analysis for: {original_prompt}\n")
        
        for response in responses:
            combined_parts.append(f"\n{response.framework.upper()} Compliance:")
            combined_parts.append(response.response)
            
            if response.required_documents:
                combined_parts.append(f"\nRequired Documents: {', '.join(response.required_documents)}")
            
            if response.evidence_requirements:
                combined_parts.append(f"\nEvidence Requirements: {', '.join(response.evidence_requirements)}")
        
        # Add summary
        frameworks_used = [r.framework for r in responses]
        combined_parts.append(f"\n\nSummary: This analysis covers {', '.join(frameworks_used)} compliance requirements.")
        
        return "\n".join(combined_parts)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all registered models."""
        status = {}
        
        for framework, model_info in self.models.items():
            status[framework] = {
                "framework": framework,
                "description": model_info.description,
                "is_loaded": model_info.is_loaded,
                "model_path": model_info.model_path,
                "keywords": model_info.keywords
            }
        
        return status
    
    def cleanup(self):
        """Clean up resources."""
        for framework in list(self.models.keys()):
            self.unload_model(framework)
        
        logger.info("Broker cleanup completed") 