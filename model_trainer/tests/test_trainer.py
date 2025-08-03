"""
Unit tests for the Model Trainer application.

This module contains comprehensive tests for the model trainer functionality,
including data preparation, model training, and evaluation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from src.trainer import ModelTrainer, TrainingConfig, TrainingMetrics


class TestTrainingConfig:
    """Test cases for TrainingConfig dataclass."""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig with default values."""
        config = TrainingConfig()
        
        assert config.model_name == "distilbert-base-uncased"
        assert config.regulatory_framework == "sox"
        assert config.max_length == 512
        assert config.batch_size == 8
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
    
    def test_training_config_custom(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            model_name="bert-base-uncased",
            regulatory_framework="fda",
            max_length=256,
            batch_size=16,
            learning_rate=1e-5,
            num_epochs=5
        )
        
        assert config.model_name == "bert-base-uncased"
        assert config.regulatory_framework == "fda"
        assert config.max_length == 256
        assert config.batch_size == 16
        assert config.learning_rate == 1e-5
        assert config.num_epochs == 5


class TestTrainingMetrics:
    """Test cases for TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test TrainingMetrics creation."""
        metrics = TrainingMetrics(
            framework="sox",
            model_name="distilbert-base-uncased",
            training_time=120.5,
            final_loss=0.15,
            final_accuracy=0.85,
            best_epoch=2,
            total_steps=1000,
            learning_rate=2e-5,
            batch_size=8,
            dataset_size=1000
        )
        
        assert metrics.framework == "sox"
        assert metrics.training_time == 120.5
        assert metrics.final_loss == 0.15
        assert metrics.final_accuracy == 0.85
        assert metrics.best_epoch == 2


class TestModelTrainer:
    """Test cases for the ModelTrainer class."""
    
    @pytest.fixture
    def config(self):
        """Create a training config for testing."""
        return TrainingConfig(
            model_name="distilbert-base-uncased",
            regulatory_framework="sox",
            max_length=128,
            batch_size=4,
            learning_rate=2e-5,
            num_epochs=1
        )
    
    @pytest.fixture
    def trainer(self, config):
        """Create a model trainer instance for testing."""
        return ModelTrainer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data for testing."""
        return [
            {
                "text_content": "SOX Section 404 requires internal controls for financial reporting.",
                "regulatory_framework": "sox",
                "document_type": "regulation",
                "page_count": 5,
                "confidence_score": 0.9
            },
            {
                "text_content": "Internal control documentation must include control environment assessment.",
                "regulatory_framework": "sox",
                "document_type": "compliance_document",
                "page_count": 3,
                "confidence_score": 0.8
            }
        ]
    
    def test_trainer_initialization(self, trainer, config):
        """Test ModelTrainer initialization."""
        assert trainer.config == config
        assert trainer.tokenizer is None
        assert trainer.model is None
        assert trainer.trainer is None
        assert "sox" in trainer.framework_configs
        assert "gaap" in trainer.framework_configs
        assert "fda" in trainer.framework_configs
    
    def test_load_processed_data_json(self, trainer):
        """Test loading processed data from JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample JSON files
            data_dir = Path(temp_dir)
            data_dir.mkdir()
            
            sample_data = [
                {
                    "text_content": "Sample SOX content",
                    "regulatory_framework": "sox",
                    "document_type": "regulation"
                }
            ]
            
            json_file = data_dir / "sample.json"
            with open(json_file, 'w') as f:
                json.dump(sample_data[0], f)
            
            loaded_data = trainer._load_processed_data(str(data_dir))
            
            assert len(loaded_data) == 1
            assert loaded_data[0]["text_content"] == "Sample SOX content"
            assert loaded_data[0]["regulatory_framework"] == "sox"
    
    def test_load_processed_data_csv(self, trainer):
        """Test loading processed data from CSV files."""
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV file
            data_dir = Path(temp_dir)
            data_dir.mkdir()
            
            sample_data = pd.DataFrame([
                {
                    "text_content": "Sample SOX content",
                    "regulatory_framework": "sox",
                    "document_type": "regulation"
                }
            ])
            
            csv_file = data_dir / "sample.csv"
            sample_data.to_csv(csv_file, index=False)
            
            loaded_data = trainer._load_processed_data(str(data_dir))
            
            assert len(loaded_data) == 1
            assert loaded_data[0]["text_content"] == "Sample SOX content"
            assert loaded_data[0]["regulatory_framework"] == "sox"
    
    def test_filter_by_framework(self, trainer, sample_data):
        """Test filtering data by regulatory framework."""
        # Add some non-SOX data
        mixed_data = sample_data + [
            {
                "text_content": "GAAP accounting standards",
                "regulatory_framework": "gaap",
                "document_type": "regulation"
            }
        ]
        
        filtered_data = trainer._filter_by_framework(mixed_data)
        
        assert len(filtered_data) == 2
        for item in filtered_data:
            assert item["regulatory_framework"] == "sox"
    
    def test_create_training_examples(self, trainer, sample_data):
        """Test creating training examples from processed data."""
        examples = trainer._create_training_examples(sample_data)
        
        assert len(examples) > 0
        for example in examples:
            assert "prompt" in example
            assert "response" in example
            assert "text" in example
            assert "SOX" in example["prompt"] or "compliance" in example["prompt"]
    
    def test_create_prompt_response_pairs_sox(self, trainer):
        """Test creating prompt-response pairs for SOX framework."""
        text_content = "SOX Section 404 requires internal controls for financial reporting."
        metadata = {"document_type": "regulation"}
        
        pairs = trainer._create_prompt_response_pairs(text_content, metadata)
        
        assert len(pairs) > 0
        for pair in pairs:
            assert "SOX" in pair["prompt"]
            assert "internal control" in pair["response"] or "documentation" in pair["response"]
    
    def test_create_prompt_response_pairs_gaap(self):
        """Test creating prompt-response pairs for GAAP framework."""
        config = TrainingConfig(regulatory_framework="gaap")
        trainer = ModelTrainer(config)
        
        text_content = "GAAP requires revenue recognition based on ASC 606."
        metadata = {"document_type": "regulation"}
        
        pairs = trainer._create_prompt_response_pairs(text_content, metadata)
        
        assert len(pairs) > 0
        for pair in pairs:
            assert "GAAP" in pair["prompt"]
            assert "accounting" in pair["response"] or "financial" in pair["response"]
    
    def test_extract_key_information(self, trainer):
        """Test extracting key information from text content."""
        text_content = "SOX Section 404 requires internal controls for financial reporting. This includes control environment assessment and risk evaluation."
        
        key_info = trainer._extract_key_information(text_content)
        
        assert "SOX Section 404" in key_info
        assert len(key_info) > 0
    
    def test_generate_structured_response_sox(self, trainer):
        """Test generating structured response for SOX framework."""
        text_content = "SOX compliance requirements for internal controls."
        metadata = {"document_type": "regulation"}
        
        response = trainer._generate_structured_response(text_content, metadata)
        
        assert "internal controls" in response
        assert "documentation" in response
        assert "evidence" in response
    
    def test_generate_structured_response_gaap(self):
        """Test generating structured response for GAAP framework."""
        config = TrainingConfig(regulatory_framework="gaap")
        trainer = ModelTrainer(config)
        
        text_content = "GAAP accounting standards for financial reporting."
        metadata = {"document_type": "regulation"}
        
        response = trainer._generate_structured_response(text_content, metadata)
        
        assert "accounting" in response
        assert "financial" in response
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_setup_model(self, mock_model, mock_tokenizer, trainer):
        """Test model and tokenizer setup."""
        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.add_special_tokens = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.resize_token_embeddings = Mock()
        mock_model_instance.to = Mock()
        mock_model.return_value = mock_model_instance
        
        trainer.setup_model()
        
        assert trainer.tokenizer is not None
        assert trainer.model is not None
        mock_tokenizer_instance.add_special_tokens.assert_called_once()
        mock_model_instance.resize_token_embeddings.assert_called_once()
        mock_model_instance.to.assert_called_once()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenize_data(self, mock_tokenizer, trainer):
        """Test data tokenization."""
        from datasets import Dataset
        
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.side_effect = lambda **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        trainer.tokenizer = mock_tokenizer_instance
        
        # Create sample dataset
        dataset = Dataset.from_dict({
            "text": ["Sample text 1", "Sample text 2"]
        })
        
        tokenized_dataset = trainer.tokenize_data(dataset)
        
        assert tokenized_dataset is not None
        assert "input_ids" in tokenized_dataset.column_names
    
    @patch('transformers.TrainingArguments')
    @patch('transformers.Trainer')
    @patch('transformers.DataCollatorForLanguageModeling')
    def test_train(self, mock_collator, mock_trainer_class, mock_args, trainer):
        """Test model training."""
        from datasets import Dataset
        
        # Mock training components
        mock_args_instance = Mock()
        mock_args.return_value = mock_args_instance
        
        mock_collator_instance = Mock()
        mock_collator.return_value = mock_collator_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = Mock(global_step=100)
        mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.15}
        mock_trainer_instance.state.best_metric = 2
        mock_trainer_class.return_value = mock_trainer_instance
        
        # Mock tokenizer and model
        trainer.tokenizer = Mock()
        trainer.model = Mock()
        
        # Create sample datasets
        train_dataset = Dataset.from_dict({"text": ["Sample text"] * 10})
        val_dataset = Dataset.from_dict({"text": ["Sample text"] * 5})
        
        # Train model
        metrics = trainer.train(train_dataset, val_dataset)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.framework == "sox"
        assert metrics.final_loss == 0.15
        assert metrics.final_accuracy == 0.85  # 1.0 - 0.15
        assert metrics.total_steps == 100
    
    def test_save_model(self, trainer):
        """Test model saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_model"
            
            # Mock model and tokenizer
            trainer.model = Mock()
            trainer.model.save_pretrained = Mock()
            
            trainer.tokenizer = Mock()
            trainer.tokenizer.save_pretrained = Mock()
            
            trainer.save_model(str(output_path))
            
            trainer.model.save_pretrained.assert_called_once_with(output_path)
            trainer.tokenizer.save_pretrained.assert_called_once_with(output_path)
            
            # Check if config file was created
            config_file = output_path / "training_config.json"
            assert config_file.exists()
    
    def test_generate_response(self, trainer):
        """Test response generation."""
        # Mock tokenizer and model
        trainer.tokenizer = Mock()
        trainer.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        trainer.tokenizer.decode.return_value = "Generated response text"
        
        trainer.model = Mock()
        trainer.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        response = trainer.generate_response("Test prompt")
        
        assert response == "Generated response text"
        trainer.tokenizer.assert_called_once()
        trainer.model.generate.assert_called_once()
    
    def test_prepare_data_no_data(self, trainer):
        """Test data preparation with no data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="No data found"):
                trainer.prepare_data(temp_dir)
    
    def test_prepare_data_no_framework_data(self, trainer):
        """Test data preparation with no framework-specific data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data for different framework
            data_dir = Path(temp_dir)
            data_dir.mkdir()
            
            sample_data = {
                "text_content": "GAAP accounting standards",
                "regulatory_framework": "gaap",
                "document_type": "regulation"
            }
            
            json_file = data_dir / "sample.json"
            with open(json_file, 'w') as f:
                json.dump(sample_data, f)
            
            with pytest.raises(ValueError, match="No data found for framework"):
                trainer.prepare_data(temp_dir)
    
    def test_prepare_data_success(self, trainer, sample_data):
        """Test successful data preparation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data files
            data_dir = Path(temp_dir)
            data_dir.mkdir()
            
            for i, data in enumerate(sample_data):
                json_file = data_dir / f"sample_{i}.json"
                with open(json_file, 'w') as f:
                    json.dump(data, f)
            
            train_dataset, val_dataset, test_dataset = trainer.prepare_data(temp_dir)
            
            assert train_dataset is not None
            assert val_dataset is not None
            assert test_dataset is not None
            assert len(train_dataset) > 0
            assert len(val_dataset) > 0
            assert len(test_dataset) > 0


class TestIntegration:
    """Integration tests for the model trainer."""
    
    @pytest.mark.integration
    def test_full_training_pipeline(self):
        """Test the complete training pipeline."""
        config = TrainingConfig(
            model_name="distilbert-base-uncased",
            regulatory_framework="sox",
            max_length=64,
            batch_size=2,
            learning_rate=2e-5,
            num_epochs=1
        )
        
        trainer = ModelTrainer(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample training data
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()
            
            sample_data = {
                "text_content": "SOX Section 404 requires internal controls for financial reporting.",
                "regulatory_framework": "sox",
                "document_type": "regulation",
                "page_count": 5,
                "confidence_score": 0.9
            }
            
            json_file = data_dir / "sample.json"
            with open(json_file, 'w') as f:
                json.dump(sample_data, f)
            
            # Prepare data
            train_dataset, val_dataset, test_dataset = trainer.prepare_data(str(data_dir))
            
            # This would require actual model training, so we'll just verify data preparation
            assert len(train_dataset) > 0
            assert len(val_dataset) > 0
            assert len(test_dataset) > 0 