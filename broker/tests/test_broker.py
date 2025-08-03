"""
Unit tests for the Compliance Broker application.

This module contains comprehensive tests for the broker functionality,
including model management, request processing, and response generation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import torch
import numpy as np

from src.broker import (
    ComplianceBroker, ModelInfo, TaskRequest, TaskResponse, 
    BrokerResponse
)


class TestModelInfo:
    """Test cases for ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation."""
        model_info = ModelInfo(
            framework="sox",
            model_path="/path/to/model",
            tokenizer_path="/path/to/tokenizer",
            description="SOX compliance model",
            keywords=["sox", "internal control", "audit"]
        )
        
        assert model_info.framework == "sox"
        assert model_info.model_path == "/path/to/model"
        assert model_info.tokenizer_path == "/path/to/tokenizer"
        assert model_info.description == "SOX compliance model"
        assert len(model_info.keywords) == 3
        assert not model_info.is_loaded
        assert model_info.model is None
        assert model_info.tokenizer is None


class TestTaskRequest:
    """Test cases for TaskRequest dataclass."""
    
    def test_task_request_creation(self):
        """Test TaskRequest creation."""
        request = TaskRequest(
            prompt="Generate SOX compliance documentation",
            frameworks=["sox"],
            max_response_length=500,
            temperature=0.7,
            include_evidence=True,
            include_documents=True
        )
        
        assert request.prompt == "Generate SOX compliance documentation"
        assert request.frameworks == ["sox"]
        assert request.max_response_length == 500
        assert request.temperature == 0.7
        assert request.include_evidence
        assert request.include_documents
    
    def test_task_request_defaults(self):
        """Test TaskRequest with default values."""
        request = TaskRequest(prompt="Test prompt")
        
        assert request.prompt == "Test prompt"
        assert request.frameworks is None
        assert request.max_response_length == 500
        assert request.temperature == 0.7
        assert request.include_evidence
        assert request.include_documents


class TestTaskResponse:
    """Test cases for TaskResponse dataclass."""
    
    def test_task_response_creation(self):
        """Test TaskResponse creation."""
        response = TaskResponse(
            framework="sox",
            response="Generated compliance documentation",
            confidence=0.85,
            required_documents=["control documentation", "audit report"],
            evidence_requirements=["audit trail", "control testing"],
            processing_time=2.5,
            model_used="/path/to/model"
        )
        
        assert response.framework == "sox"
        assert response.response == "Generated compliance documentation"
        assert response.confidence == 0.85
        assert len(response.required_documents) == 2
        assert len(response.evidence_requirements) == 2
        assert response.processing_time == 2.5
        assert response.model_used == "/path/to/model"


class TestBrokerResponse:
    """Test cases for BrokerResponse dataclass."""
    
    def test_broker_response_creation(self):
        """Test BrokerResponse creation."""
        task_responses = [
            TaskResponse(
                framework="sox",
                response="SOX compliance response",
                confidence=0.8,
                required_documents=[],
                evidence_requirements=[],
                processing_time=1.0,
                model_used="/path/to/sox_model"
            )
        ]
        
        response = BrokerResponse(
            original_prompt="Generate compliance documentation",
            responses=task_responses,
            combined_response="Combined compliance response",
            total_processing_time=3.0,
            frameworks_used=["sox"],
            overall_confidence=0.8
        )
        
        assert response.original_prompt == "Generate compliance documentation"
        assert len(response.responses) == 1
        assert response.combined_response == "Combined compliance response"
        assert response.total_processing_time == 3.0
        assert response.frameworks_used == ["sox"]
        assert response.overall_confidence == 0.8


class TestComplianceBroker:
    """Test cases for the ComplianceBroker class."""
    
    @pytest.fixture
    def broker(self):
        """Create a broker instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return ComplianceBroker(models_dir=temp_dir)
    
    def test_broker_initialization(self, broker):
        """Test ComplianceBroker initialization."""
        assert broker.device is not None
        assert len(broker.models) == 0
        assert "sox" in broker.framework_configs
        assert "gaap" in broker.framework_configs
        assert "fda" in broker.framework_configs
        assert "hipaa" in broker.framework_configs
        assert "gdpr" in broker.framework_configs
    
    def test_register_model_success(self, broker):
        """Test successful model registration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sox_model"
            model_path.mkdir()
            
            # Create dummy model files
            (model_path / "pytorch_model.bin").touch()
            (model_path / "tokenizer.json").touch()
            
            success = broker.register_model("sox", str(model_path))
            
            assert success
            assert "sox" in broker.models
            assert broker.models["sox"].framework == "sox"
            assert broker.models["sox"].model_path == str(model_path)
    
    def test_register_model_invalid_framework(self, broker):
        """Test model registration with invalid framework."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "invalid_model"
            model_path.mkdir()
            
            success = broker.register_model("invalid", str(model_path))
            
            assert not success
    
    def test_register_model_nonexistent_path(self, broker):
        """Test model registration with nonexistent path."""
        success = broker.register_model("sox", "/nonexistent/path")
        
        assert not success
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_model_success(self, mock_model, mock_tokenizer, broker):
        """Test successful model loading."""
        # Register a model first
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sox_model"
            model_path.mkdir()
            (model_path / "pytorch_model.bin").touch()
            (model_path / "tokenizer.json").touch()
            
            broker.register_model("sox", str(model_path))
            
            # Mock tokenizer and model
            mock_tokenizer_instance = Mock()
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            mock_model_instance = Mock()
            mock_model_instance.to = Mock()
            mock_model_instance.eval = Mock()
            mock_model.return_value = mock_model_instance
            
            success = broker.load_model("sox")
            
            assert success
            assert broker.models["sox"].is_loaded
            assert broker.models["sox"].model is not None
            assert broker.models["sox"].tokenizer is not None
    
    def test_load_model_not_registered(self, broker):
        """Test loading unregistered model."""
        success = broker.load_model("nonexistent")
        
        assert not success
    
    def test_unload_model_success(self, broker):
        """Test successful model unloading."""
        # Register and load a model first
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sox_model"
            model_path.mkdir()
            (model_path / "pytorch_model.bin").touch()
            (model_path / "tokenizer.json").touch()
            
            broker.register_model("sox", str(model_path))
            
            # Mock the model as loaded
            broker.models["sox"].is_loaded = True
            broker.models["sox"].model = Mock()
            broker.models["sox"].tokenizer = Mock()
            
            success = broker.unload_model("sox")
            
            assert success
            assert not broker.models["sox"].is_loaded
            assert broker.models["sox"].model is None
            assert broker.models["sox"].tokenizer is None
    
    def test_detect_relevant_frameworks_sox(self, broker):
        """Test SOX framework detection."""
        prompt = "Generate SOX-compliant internal control documentation"
        frameworks = broker.detect_relevant_frameworks(prompt)
        
        assert "sox" in frameworks
    
    def test_detect_relevant_frameworks_gaap(self, broker):
        """Test GAAP framework detection."""
        prompt = "Create GAAP-compliant financial statements"
        frameworks = broker.detect_relevant_frameworks(prompt)
        
        assert "gaap" in frameworks
    
    def test_detect_relevant_frameworks_fda(self, broker):
        """Test FDA framework detection."""
        prompt = "Generate FDA 21 CFR Part 11 compliance documentation"
        frameworks = broker.detect_relevant_frameworks(prompt)
        
        assert "fda" in frameworks
    
    def test_detect_relevant_frameworks_multiple(self, broker):
        """Test detection of multiple frameworks."""
        prompt = "Ensure SOX and GAAP compliance for financial reporting"
        frameworks = broker.detect_relevant_frameworks(prompt)
        
        assert "sox" in frameworks
        assert "gaap" in frameworks
    
    def test_detect_relevant_frameworks_none(self, broker):
        """Test detection when no frameworks are relevant."""
        prompt = "Generate general documentation"
        frameworks = broker.detect_relevant_frameworks(prompt)
        
        # Should return all available frameworks when none detected
        assert len(frameworks) == 0
    
    @pytest.mark.asyncio
    async def test_process_request_single_framework(self, broker):
        """Test processing request with single framework."""
        # Register and load a model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sox_model"
            model_path.mkdir()
            (model_path / "pytorch_model.bin").touch()
            (model_path / "tokenizer.json").touch()
            
            broker.register_model("sox", str(model_path))
            
            # Mock model loading
            broker.models["sox"].is_loaded = True
            broker.models["sox"].model = Mock()
            broker.models["sox"].tokenizer = Mock()
            
            # Mock response generation
            broker.models["sox"].tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])
            }
            broker.models["sox"].tokenizer.decode.return_value = "Generated SOX response"
            broker.models["sox"].model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            
            request = TaskRequest(
                prompt="Generate SOX compliance documentation",
                frameworks=["sox"]
            )
            
            response = await broker.process_request(request)
            
            assert response.original_prompt == "Generate SOX compliance documentation"
            assert len(response.responses) == 1
            assert response.responses[0].framework == "sox"
            assert "SOX" in response.combined_response
            assert response.frameworks_used == ["sox"]
    
    @pytest.mark.asyncio
    async def test_process_request_multiple_frameworks(self, broker):
        """Test processing request with multiple frameworks."""
        # Register models for multiple frameworks
        with tempfile.TemporaryDirectory() as temp_dir:
            for framework in ["sox", "gaap"]:
                model_path = Path(temp_dir) / f"{framework}_model"
                model_path.mkdir()
                (model_path / "pytorch_model.bin").touch()
                (model_path / "tokenizer.json").touch()
                
                broker.register_model(framework, str(model_path))
                
                # Mock model loading
                broker.models[framework].is_loaded = True
                broker.models[framework].model = Mock()
                broker.models[framework].tokenizer = Mock()
                
                # Mock response generation
                broker.models[framework].tokenizer.return_value = {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])
                }
                broker.models[framework].tokenizer.decode.return_value = f"Generated {framework.upper()} response"
                broker.models[framework].model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            
            request = TaskRequest(
                prompt="Generate SOX and GAAP compliance documentation",
                frameworks=["sox", "gaap"]
            )
            
            response = await broker.process_request(request)
            
            assert len(response.responses) == 2
            assert response.frameworks_used == ["sox", "gaap"]
            assert "SOX" in response.combined_response
            assert "GAAP" in response.combined_response
    
    def test_generate_response(self, broker):
        """Test response generation."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.decode.return_value = "Generated response text"
        
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        response = broker._generate_response(
            mock_model, mock_tokenizer, "Test prompt", max_length=200, temperature=0.7
        )
        
        assert response == "Generated response text"
        mock_tokenizer.assert_called_once()
        mock_model.generate.assert_called_once()
    
    def test_extract_required_documents_sox(self, broker):
        """Test extraction of required documents for SOX."""
        response_text = "Required documents include control documentation and audit reports."
        documents = broker._extract_required_documents(response_text, "sox")
        
        assert "control documentation" in documents
        assert "audit report" in documents
    
    def test_extract_required_documents_gaap(self, broker):
        """Test extraction of required documents for GAAP."""
        response_text = "Financial statements and accounting policies are required."
        documents = broker._extract_required_documents(response_text, "gaap")
        
        assert "financial statement" in documents
        assert "accounting policy" in documents
    
    def test_extract_evidence_requirements_sox(self, broker):
        """Test extraction of evidence requirements for SOX."""
        response_text = "Evidence must include audit trails and control testing results."
        evidence = broker._extract_evidence_requirements(response_text, "sox")
        
        assert "audit trail" in evidence
        assert "control testing" in evidence
    
    def test_calculate_confidence(self, broker):
        """Test confidence calculation."""
        # Test with good response
        good_response = "SOX compliance requires internal controls for financial reporting. Documentation must include control environment assessment and evidence of testing."
        confidence = broker._calculate_confidence(good_response, "sox")
        
        assert confidence > 0.5
        
        # Test with poor response
        poor_response = "Some text."
        confidence = broker._calculate_confidence(poor_response, "sox")
        
        assert confidence < 0.5
    
    def test_combine_responses_single(self, broker):
        """Test combining single response."""
        responses = [
            TaskResponse(
                framework="sox",
                response="SOX compliance response",
                confidence=0.8,
                required_documents=["doc1"],
                evidence_requirements=["evidence1"],
                processing_time=1.0,
                model_used="/path/to/model"
            )
        ]
        
        combined = broker._combine_responses(responses, "Test prompt")
        
        assert "SOX compliance response" in combined
        assert "Test prompt" in combined
    
    def test_combine_responses_multiple(self, broker):
        """Test combining multiple responses."""
        responses = [
            TaskResponse(
                framework="sox",
                response="SOX compliance response",
                confidence=0.8,
                required_documents=["sox_doc"],
                evidence_requirements=["sox_evidence"],
                processing_time=1.0,
                model_used="/path/to/sox_model"
            ),
            TaskResponse(
                framework="gaap",
                response="GAAP compliance response",
                confidence=0.7,
                required_documents=["gaap_doc"],
                evidence_requirements=["gaap_evidence"],
                processing_time=1.5,
                model_used="/path/to/gaap_model"
            )
        ]
        
        combined = broker._combine_responses(responses, "Test prompt")
        
        assert "SOX compliance response" in combined
        assert "GAAP compliance response" in combined
        assert "sox_doc" in combined
        assert "gaap_doc" in combined
        assert "sox, gaap" in combined.lower()
    
    def test_combine_responses_empty(self, broker):
        """Test combining empty responses."""
        combined = broker._combine_responses([], "Test prompt")
        
        assert "No relevant models available" in combined
    
    def test_get_model_status(self, broker):
        """Test getting model status."""
        # Register a model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sox_model"
            model_path.mkdir()
            (model_path / "pytorch_model.bin").touch()
            (model_path / "tokenizer.json").touch()
            
            broker.register_model("sox", str(model_path))
            
            status = broker.get_model_status()
            
            assert "sox" in status
            assert status["sox"]["framework"] == "sox"
            assert status["sox"]["is_loaded"] == False
            assert status["sox"]["model_path"] == str(model_path)
    
    def test_cleanup(self, broker):
        """Test broker cleanup."""
        # Register and load a model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sox_model"
            model_path.mkdir()
            (model_path / "pytorch_model.bin").touch()
            (model_path / "tokenizer.json").touch()
            
            broker.register_model("sox", str(model_path))
            broker.models["sox"].is_loaded = True
            broker.models["sox"].model = Mock()
            broker.models["sox"].tokenizer = Mock()
            
            broker.cleanup()
            
            assert not broker.models["sox"].is_loaded
            assert broker.models["sox"].model is None
            assert broker.models["sox"].tokenizer is None


class TestIntegration:
    """Integration tests for the broker."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_request_processing(self):
        """Test full request processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            broker = ComplianceBroker(models_dir=temp_dir)
            
            # Register models for multiple frameworks
            for framework in ["sox", "gaap"]:
                model_path = Path(temp_dir) / f"{framework}_model"
                model_path.mkdir()
                (model_path / "pytorch_model.bin").touch()
                (model_path / "tokenizer.json").touch()
                
                broker.register_model(framework, str(model_path))
                
                # Mock model loading and response generation
                broker.models[framework].is_loaded = True
                broker.models[framework].model = Mock()
                broker.models[framework].tokenizer = Mock()
                
                broker.models[framework].tokenizer.return_value = {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])
                }
                broker.models[framework].tokenizer.decode.return_value = f"Generated {framework.upper()} compliance documentation"
                broker.models[framework].model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            
            # Process request
            request = TaskRequest(
                prompt="Generate comprehensive compliance documentation for financial reporting",
                frameworks=["sox", "gaap"]
            )
            
            response = await broker.process_request(request)
            
            # Verify response
            assert response.original_prompt == request.prompt
            assert len(response.responses) == 2
            assert response.frameworks_used == ["sox", "gaap"]
            assert response.overall_confidence > 0
            assert response.total_processing_time > 0
            
            # Verify individual responses
            for task_response in response.responses:
                assert task_response.framework in ["sox", "gaap"]
                assert task_response.confidence > 0
                assert task_response.processing_time > 0
                assert len(task_response.required_documents) >= 0
                assert len(task_response.evidence_requirements) >= 0 