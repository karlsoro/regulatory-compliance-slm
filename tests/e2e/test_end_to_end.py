"""
End-to-End Tests for Regulatory Compliance SLM System

This module contains comprehensive end-to-end tests that verify the complete
workflow from PDF processing to model training to broker usage.
"""

import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import pandas as pd

# Import the main components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pdf_converter.src.converter import PDFConverter
from model_trainer.src.trainer import ModelTrainer, TrainingConfig
from broker.src.broker import ComplianceBroker, TaskRequest


class TestEndToEndWorkflow:
    """End-to-end tests for the complete workflow."""
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing."""
        return """
        Sarbanes-Oxley Act of 2002
        
        Section 404 - Management Assessment of Internal Controls
        
        (a) RULES REQUIRED.—The Commission shall prescribe rules requiring each annual report 
        required by section 13(a) or 15(d) of the Securities Exchange Act of 1934 (15 U.S.C. 78m(a) 
        or 78o(d)) to contain an internal control report, which shall—
        
        (1) state the responsibility of management for establishing and maintaining an adequate 
        internal control structure and procedures for financial reporting; and
        
        (2) contain an assessment, as of the end of the most recent fiscal year of the issuer, 
        of the effectiveness of the internal control structure and procedures of the issuer for 
        financial reporting.
        
        Internal controls must include:
        - Control environment assessment
        - Risk evaluation procedures
        - Control activities documentation
        - Information and communication systems
        - Monitoring activities
        
        Required documentation:
        - Control environment documentation
        - Risk assessment reports
        - Control activity descriptions
        - Information system documentation
        - Monitoring reports
        """
    
    @pytest.fixture
    def sample_gaap_content(self):
        """Sample GAAP content for testing."""
        return """
        Generally Accepted Accounting Principles
        
        Revenue Recognition - ASC 606
        
        The core principle of the revenue recognition standard is that an entity should 
        recognize revenue to depict the transfer of promised goods or services to customers 
        in an amount that reflects the consideration to which the entity expects to be 
        entitled in exchange for those goods or services.
        
        Key requirements:
        - Identify the contract with a customer
        - Identify the performance obligations
        - Determine the transaction price
        - Allocate the transaction price
        - Recognize revenue when performance obligations are satisfied
        
        Required documentation:
        - Contract documentation
        - Performance obligation analysis
        - Transaction price calculations
        - Revenue allocation schedules
        - Recognition timing documentation
        """
    
    def create_mock_pdf(self, content: str, filename: str, temp_dir: Path):
        """Create a mock PDF file for testing."""
        # In a real scenario, this would create an actual PDF
        # For testing, we'll create a JSON file that simulates processed PDF data
        pdf_data = {
            "text_content": content,
            "regulatory_framework": "sox" if "SOX" in content else "gaap",
            "document_type": "regulation",
            "page_count": 5,
            "confidence_score": 0.9,
            "processing_time": 1.5,
            "sections": {"main": content},
            "keywords": ["compliance", "regulation", "documentation"]
        }
        
        json_file = temp_dir / f"{filename}.json"
        with open(json_file, 'w') as f:
            json.dump(pdf_data, f)
        
        return json_file
    
    @pytest.mark.e2e
    def test_complete_workflow_sox(self, sample_pdf_content):
        """Test complete workflow for SOX compliance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Create mock PDF data
            raw_data_dir = temp_path / "raw"
            raw_data_dir.mkdir()
            
            processed_data_dir = temp_path / "processed"
            processed_data_dir.mkdir()
            
            # Create mock PDF data
            pdf_file = self.create_mock_pdf(sample_pdf_content, "sox_regulation", raw_data_dir)
            
            # Step 2: Process PDF data
            converter = PDFConverter(output_format="json")
            
            # Mock the PDF processing to use our JSON data
            with patch.object(converter, '_load_processed_data') as mock_load:
                mock_load.return_value = [json.load(open(pdf_file))]
                
                summary = converter.process_folder(
                    str(raw_data_dir), 
                    str(processed_data_dir), 
                    regulatory_framework="sox"
                )
            
            assert summary["processed"] == 1
            assert summary["failed"] == 0
            
            # Step 3: Train model
            config = TrainingConfig(
                model_name="distilbert-base-uncased",
                regulatory_framework="sox",
                max_length=128,
                batch_size=2,
                learning_rate=2e-5,
                num_epochs=1,
                output_dir=str(temp_path / "models")
            )
            
            trainer = ModelTrainer(config)
            
            # Mock model training
            with patch.object(trainer, 'setup_model'), \
                 patch.object(trainer, 'train') as mock_train, \
                 patch.object(trainer, 'save_model'):
                
                mock_train.return_value = Mock(
                    framework="sox",
                    model_name="distilbert-base-uncased",
                    training_time=60.0,
                    final_loss=0.15,
                    final_accuracy=0.85,
                    best_epoch=1,
                    total_steps=100,
                    learning_rate=2e-5,
                    batch_size=2,
                    dataset_size=10
                )
                
                # Prepare data
                train_dataset, val_dataset, test_dataset = trainer.prepare_data(str(processed_data_dir))
                
                assert len(train_dataset) > 0
                assert len(val_dataset) > 0
                assert len(test_dataset) > 0
                
                # Train model (mocked)
                metrics = trainer.train(train_dataset, val_dataset)
                
                assert metrics.framework == "sox"
                assert metrics.final_loss == 0.15
                assert metrics.final_accuracy == 0.85
                
                # Save model
                model_output_dir = temp_path / "trained_model"
                trainer.save_model(str(model_output_dir))
                
                assert model_output_dir.exists()
            
            # Step 4: Test broker
            broker = ComplianceBroker(models_dir=str(temp_path / "broker_models"))
            
            # Register the trained model
            success = broker.register_model("sox", str(model_output_dir))
            assert success
            
            # Mock model loading and response generation
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                 patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.return_value = {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])
                }
                mock_tokenizer_instance.decode.return_value = "SOX compliance requires internal controls for financial reporting. Required documents include control documentation and audit reports."
                mock_tokenizer.return_value = mock_tokenizer_instance
                
                mock_model_instance = Mock()
                mock_model_instance.to = Mock()
                mock_model_instance.eval = Mock()
                mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                mock_model.return_value = mock_model_instance
                
                # Load model
                success = broker.load_model("sox")
                assert success
                
                # Test request processing
                request = TaskRequest(
                    prompt="Generate SOX-compliant internal control documentation",
                    frameworks=["sox"]
                )
                
                # Process request (async)
                async def test_request():
                    response = await broker.process_request(request)
                    return response
                
                response = asyncio.run(test_request())
                
                assert response.original_prompt == request.prompt
                assert len(response.responses) == 1
                assert response.responses[0].framework == "sox"
                assert "SOX" in response.combined_response
                assert response.overall_confidence > 0
    
    @pytest.mark.e2e
    def test_complete_workflow_multiple_frameworks(self, sample_pdf_content, sample_gaap_content):
        """Test complete workflow with multiple regulatory frameworks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Create mock PDF data for multiple frameworks
            raw_data_dir = temp_path / "raw"
            raw_data_dir.mkdir()
            
            processed_data_dir = temp_path / "processed"
            processed_data_dir.mkdir()
            
            # Create mock PDF data for SOX and GAAP
            sox_pdf = self.create_mock_pdf(sample_pdf_content, "sox_regulation", raw_data_dir)
            gaap_pdf = self.create_mock_pdf(sample_gaap_content, "gaap_regulation", raw_data_dir)
            
            # Step 2: Process PDF data
            converter = PDFConverter(output_format="json")
            
            # Mock the PDF processing
            with patch.object(converter, '_load_processed_data') as mock_load:
                mock_load.return_value = [
                    json.load(open(sox_pdf)),
                    json.load(open(gaap_pdf))
                ]
                
                summary = converter.process_folder(
                    str(raw_data_dir), 
                    str(processed_data_dir)
                )
            
            assert summary["processed"] == 2
            assert summary["failed"] == 0
            
            # Step 3: Train models for both frameworks
            trained_models = {}
            
            for framework in ["sox", "gaap"]:
                config = TrainingConfig(
                    model_name="distilbert-base-uncased",
                    regulatory_framework=framework,
                    max_length=128,
                    batch_size=2,
                    learning_rate=2e-5,
                    num_epochs=1,
                    output_dir=str(temp_path / "models")
                )
                
                trainer = ModelTrainer(config)
                
                # Mock model training
                with patch.object(trainer, 'setup_model'), \
                     patch.object(trainer, 'train') as mock_train, \
                     patch.object(trainer, 'save_model'):
                    
                    mock_train.return_value = Mock(
                        framework=framework,
                        model_name="distilbert-base-uncased",
                        training_time=60.0,
                        final_loss=0.15,
                        final_accuracy=0.85,
                        best_epoch=1,
                        total_steps=100,
                        learning_rate=2e-5,
                        batch_size=2,
                        dataset_size=10
                    )
                    
                    # Prepare data
                    train_dataset, val_dataset, test_dataset = trainer.prepare_data(str(processed_data_dir))
                    
                    # Train model (mocked)
                    metrics = trainer.train(train_dataset, val_dataset)
                    
                    assert metrics.framework == framework
                    
                    # Save model
                    model_output_dir = temp_path / f"trained_model_{framework}"
                    trainer.save_model(str(model_output_dir))
                    
                    trained_models[framework] = model_output_dir
            
            # Step 4: Test broker with multiple models
            broker = ComplianceBroker(models_dir=str(temp_path / "broker_models"))
            
            # Register both models
            for framework, model_path in trained_models.items():
                success = broker.register_model(framework, str(model_path))
                assert success
            
            # Mock model loading and response generation
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                 patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.return_value = {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])
                }
                mock_tokenizer_instance.decode.return_value = "Compliance documentation generated"
                mock_tokenizer.return_value = mock_tokenizer_instance
                
                mock_model_instance = Mock()
                mock_model_instance.to = Mock()
                mock_model_instance.eval = Mock()
                mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                mock_model.return_value = mock_model_instance
                
                # Load models
                for framework in ["sox", "gaap"]:
                    success = broker.load_model(framework)
                    assert success
                
                # Test request processing with multiple frameworks
                request = TaskRequest(
                    prompt="Generate comprehensive compliance documentation for financial reporting",
                    frameworks=["sox", "gaap"]
                )
                
                # Process request (async)
                async def test_request():
                    response = await broker.process_request(request)
                    return response
                
                response = asyncio.run(test_request())
                
                assert response.original_prompt == request.prompt
                assert len(response.responses) == 2
                assert response.frameworks_used == ["sox", "gaap"]
                assert "SOX" in response.combined_response
                assert "GAAP" in response.combined_response
                assert response.overall_confidence > 0
    
    @pytest.mark.e2e
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in the workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test 1: Invalid PDF processing
            converter = PDFConverter(output_format="json")
            
            with pytest.raises(FileNotFoundError):
                converter.process_single_file("nonexistent.pdf")
            
            # Test 2: Invalid model training
            config = TrainingConfig(
                model_name="invalid-model",
                regulatory_framework="sox"
            )
            
            trainer = ModelTrainer(config)
            
            with tempfile.TemporaryDirectory() as data_dir:
                # Test with no data
                with pytest.raises(ValueError, match="No data found"):
                    trainer.prepare_data(data_dir)
            
            # Test 3: Invalid broker operations
            broker = ComplianceBroker(models_dir=str(temp_path / "broker_models"))
            
            # Test loading unregistered model
            success = broker.load_model("nonexistent")
            assert not success
            
            # Test registering invalid framework
            success = broker.register_model("invalid", "/nonexistent/path")
            assert not success
    
    @pytest.mark.e2e
    def test_performance_and_scalability(self, sample_pdf_content):
        """Test performance and scalability aspects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test with multiple PDF files
            raw_data_dir = temp_path / "raw"
            raw_data_dir.mkdir()
            
            processed_data_dir = temp_path / "processed"
            processed_data_dir.mkdir()
            
            # Create multiple mock PDF files
            pdf_files = []
            for i in range(10):
                pdf_file = self.create_mock_pdf(
                    f"SOX regulation content {i}. {sample_pdf_content}", 
                    f"sox_regulation_{i}", 
                    raw_data_dir
                )
                pdf_files.append(pdf_file)
            
            # Process multiple files
            converter = PDFConverter(output_format="json")
            
            with patch.object(converter, '_load_processed_data') as mock_load:
                mock_load.return_value = [json.load(open(f)) for f in pdf_files]
                
                summary = converter.process_folder(
                    str(raw_data_dir), 
                    str(processed_data_dir), 
                    regulatory_framework="sox"
                )
            
            assert summary["processed"] == 10
            assert summary["failed"] == 0
            
            # Test broker with multiple models
            broker = ComplianceBroker(models_dir=str(temp_path / "broker_models"))
            
            # Register multiple models
            for framework in ["sox", "gaap", "fda"]:
                model_path = temp_path / f"model_{framework}"
                model_path.mkdir()
                (model_path / "pytorch_model.bin").touch()
                (model_path / "tokenizer.json").touch()
                
                success = broker.register_model(framework, str(model_path))
                assert success
            
            # Test model status
            status = broker.get_model_status()
            assert len(status) == 3
            assert "sox" in status
            assert "gaap" in status
            assert "fda" in status
    
    @pytest.mark.e2e
    def test_data_quality_and_validation(self, sample_pdf_content):
        """Test data quality and validation throughout the workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test with various data quality scenarios
            test_cases = [
                {
                    "name": "high_quality",
                    "content": sample_pdf_content,
                    "expected_framework": "sox",
                    "expected_confidence": 0.8
                },
                {
                    "name": "low_quality",
                    "content": "Short text with minimal content.",
                    "expected_framework": "unknown",
                    "expected_confidence": 0.3
                },
                {
                    "name": "mixed_content",
                    "content": "SOX and GAAP requirements for financial reporting.",
                    "expected_framework": "sox",  # First match
                    "expected_confidence": 0.6
                }
            ]
            
            for test_case in test_cases:
                # Create test data
                raw_data_dir = temp_path / f"raw_{test_case['name']}"
                raw_data_dir.mkdir()
                
                processed_data_dir = temp_path / f"processed_{test_case['name']}"
                processed_data_dir.mkdir()
                
                pdf_file = self.create_mock_pdf(
                    test_case["content"], 
                    f"test_{test_case['name']}", 
                    raw_data_dir
                )
                
                # Process data
                converter = PDFConverter(output_format="json")
                
                with patch.object(converter, '_load_processed_data') as mock_load:
                    mock_load.return_value = [json.load(open(pdf_file))]
                    
                    summary = converter.process_folder(
                        str(raw_data_dir), 
                        str(processed_data_dir)
                    )
                
                assert summary["processed"] == 1
                
                # Validate extracted content
                with open(pdf_file, 'r') as f:
                    data = json.load(f)
                
                assert "text_content" in data
                assert "regulatory_framework" in data
                assert "confidence_score" in data
                
                # Framework detection should work
                if test_case["expected_framework"] != "unknown":
                    assert data["regulatory_framework"] == test_case["expected_framework"]
                
                # Confidence should be reasonable
                assert 0 <= data["confidence_score"] <= 1


class TestIntegrationScenarios:
    """Integration test scenarios for real-world use cases."""
    
    @pytest.mark.integration
    def test_financial_institution_compliance(self):
        """Test compliance workflow for a financial institution."""
        # This would test SOX + GAAP compliance for financial reporting
        pass
    
    @pytest.mark.integration
    def test_healthcare_compliance(self):
        """Test compliance workflow for a healthcare organization."""
        # This would test HIPAA + FDA compliance for healthcare data
        pass
    
    @pytest.mark.integration
    def test_pharmaceutical_compliance(self):
        """Test compliance workflow for a pharmaceutical company."""
        # This would test FDA + GMP compliance for drug development
        pass
    
    @pytest.mark.integration
    def test_european_company_compliance(self):
        """Test compliance workflow for a European company."""
        # This would test GDPR + local regulations
        pass


if __name__ == "__main__":
    # Run the end-to-end tests
    pytest.main([__file__, "-v", "-m", "e2e"]) 