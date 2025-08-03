"""
Model Trainer for Regulatory Compliance Small Language Models

This module provides functionality to train and fine-tune small language models
specifically for regulatory compliance tasks across different frameworks.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import warnings

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str = "distilbert-base-uncased"
    regulatory_framework: str = "sox"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "models/checkpoints"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class TrainingMetrics:
    """Training metrics and results."""
    framework: str
    model_name: str
    training_time: float
    final_loss: float
    final_accuracy: float
    best_epoch: int
    total_steps: int
    learning_rate: float
    batch_size: int
    dataset_size: int


class ModelTrainer:
    """
    Trainer for small language models focused on regulatory compliance.
    
    This trainer handles the complete pipeline from data preparation to model
    training and evaluation for different regulatory frameworks.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Framework-specific configurations
        self.framework_configs = {
            "sox": {
                "task_type": "text_generation",
                "base_model": "distilbert-base-uncased",
                "max_length": 512,
                "special_tokens": ["[SOX]", "[CONTROL]", "[AUDIT]", "[REPORT]"]
            },
            "gaap": {
                "task_type": "text_generation", 
                "base_model": "distilbert-base-uncased",
                "max_length": 512,
                "special_tokens": ["[GAAP]", "[ACCOUNTING]", "[FINANCIAL]", "[STATEMENT]"]
            },
            "fda": {
                "task_type": "text_generation",
                "base_model": "distilbert-base-uncased", 
                "max_length": 512,
                "special_tokens": ["[FDA]", "[CLINICAL]", "[TRIAL]", "[APPROVAL]"]
            },
            "hipaa": {
                "task_type": "text_generation",
                "base_model": "distilbert-base-uncased",
                "max_length": 512,
                "special_tokens": ["[HIPAA]", "[PRIVACY]", "[SECURITY]", "[PATIENT]"]
            },
            "gdpr": {
                "task_type": "text_generation",
                "base_model": "distilbert-base-uncased",
                "max_length": 512,
                "special_tokens": ["[GDPR]", "[DATA]", "[PROTECTION]", "[CONSENT]"]
            }
        }
        
        logger.info(f"Model Trainer initialized for {config.regulatory_framework}")
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, data_path: str) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare training data from processed PDF documents.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info(f"Preparing data from {data_path}")
        
        # Load and combine all processed data
        all_data = self._load_processed_data(data_path)
        
        if not all_data:
            raise ValueError("No data found in the specified path")
        
        # Filter data for the specific regulatory framework
        framework_data = self._filter_by_framework(all_data)
        
        if not framework_data:
            raise ValueError(f"No data found for framework: {self.config.regulatory_framework}")
        
        # Create training examples
        training_examples = self._create_training_examples(framework_data)
        
        # Split data
        train_data, temp_data = train_test_split(
            training_examples, test_size=0.3, random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42
        )
        
        # Convert to datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        logger.info(f"Data prepared: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_dataset, val_dataset, test_dataset
    
    def _load_processed_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load all processed data files."""
        data_path = Path(data_path)
        all_data = []
        
        # Load JSON files
        json_files = list(data_path.glob("**/*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {str(e)}")
        
        # Load CSV files
        csv_files = list(data_path.glob("**/*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    all_data.append(row.to_dict())
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {str(e)}")
        
        return all_data
    
    def _filter_by_framework(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data for the specific regulatory framework."""
        framework_data = []
        
        for item in data:
            # Check if item has framework information
            if isinstance(item, dict):
                framework = item.get('regulatory_framework', '').lower()
                if framework == self.config.regulatory_framework.lower():
                    framework_data.append(item)
        
        return framework_data
    
    def _create_training_examples(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create training examples from processed data."""
        examples = []
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            text_content = item.get('text_content', '')
            if not text_content or len(text_content.strip()) < 50:
                continue
            
            # Create different types of training examples
            examples.extend(self._create_prompt_response_pairs(text_content, item))
        
        return examples
    
    def _create_prompt_response_pairs(self, text_content: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create prompt-response pairs for training."""
        examples = []
        
        # Framework-specific prompt templates
        prompt_templates = {
            "sox": [
                "Generate SOX-compliant internal control documentation for: ",
                "What documents are required for SOX Section 404 compliance regarding: ",
                "Create a compliance checklist for SOX requirements related to: ",
                "Draft an audit procedure for SOX compliance covering: "
            ],
            "gaap": [
                "Generate GAAP-compliant financial reporting documentation for: ",
                "What accounting standards apply to: ",
                "Create financial statement templates for: ",
                "Draft revenue recognition procedures for: "
            ],
            "fda": [
                "Generate FDA-compliant documentation for: ",
                "What 21 CFR Part 11 requirements apply to: ",
                "Create clinical trial documentation for: ",
                "Draft electronic record validation procedures for: "
            ],
            "hipaa": [
                "Generate HIPAA-compliant privacy documentation for: ",
                "What privacy safeguards are required for: ",
                "Create patient data protection procedures for: ",
                "Draft breach notification procedures for: "
            ],
            "gdpr": [
                "Generate GDPR-compliant data protection documentation for: ",
                "What data subject rights apply to: ",
                "Create consent management procedures for: ",
                "Draft data processing agreements for: "
            ]
        }
        
        templates = prompt_templates.get(self.config.regulatory_framework, [])
        
        # Create examples with different prompts
        for template in templates:
            # Extract key information from text content
            key_info = self._extract_key_information(text_content)
            
            if key_info:
                prompt = template + key_info
                response = self._generate_structured_response(text_content, metadata)
                
                examples.append({
                    "prompt": prompt,
                    "response": response,
                    "text": f"{prompt} {response}"
                })
        
        return examples
    
    def _extract_key_information(self, text_content: str) -> str:
        """Extract key information from text content for prompts."""
        # Simple extraction - in production, use more sophisticated NLP
        sentences = text_content.split('.')
        if len(sentences) > 0:
            return sentences[0][:100] + "..."
        return "financial system"
    
    def _generate_structured_response(self, text_content: str, metadata: Dict[str, Any]) -> str:
        """Generate structured response based on content and metadata."""
        doc_type = metadata.get('document_type', 'general')
        
        # Framework-specific response templates
        response_templates = {
            "sox": {
                "regulation": "Based on SOX requirements, the following documents are required: [DOCUMENT_LIST]. Internal controls must include: [CONTROL_LIST]. Evidence requirements: [EVIDENCE_LIST].",
                "audit_report": "Audit findings indicate: [FINDINGS]. Required corrective actions: [ACTIONS]. Compliance status: [STATUS].",
                "policy": "Policy requirements: [REQUIREMENTS]. Implementation procedures: [PROCEDURES]. Monitoring controls: [CONTROLS].",
                "compliance_document": "Compliance checklist: [CHECKLIST]. Required documentation: [DOCS]. Validation procedures: [VALIDATION]."
            }
        }
        
        templates = response_templates.get(self.config.regulatory_framework, {})
        template = templates.get(doc_type, "Compliance requirements: [REQUIREMENTS]. Documentation needed: [DOCS].")
        
        # Replace placeholders with actual content
        response = template.replace("[REQUIREMENTS]", "internal controls, financial reporting, audit trails")
        response = response.replace("[DOCS]", "control documentation, process flows, evidence logs")
        response = response.replace("[CONTROLS]", "access controls, change management, monitoring")
        
        return response
    
    def setup_model(self):
        """Set up the model and tokenizer."""
        logger.info(f"Setting up model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add special tokens for the regulatory framework
        framework_config = self.framework_configs.get(self.config.regulatory_framework, {})
        special_tokens = framework_config.get("special_tokens", [])
        
        if special_tokens:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": special_tokens
            })
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        
        # Resize token embeddings if special tokens were added
        if special_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move to device
        self.model.to(self.device)
        
        logger.info("Model and tokenizer setup complete")
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> TrainingMetrics:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training metrics
        """
        logger.info("Starting model training")
        start_time = time.time()
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Tokenize datasets
        train_tokenized = self.tokenize_data(train_dataset)
        val_tokenized = self.tokenize_data(val_dataset)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=None,  # Disable wandb/tensorboard for simplicity
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Use causal language modeling
        )
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Evaluate
        eval_result = self.trainer.evaluate()
        
        # Calculate metrics
        training_time = time.time() - start_time
        final_loss = eval_result.get("eval_loss", float('inf'))
        final_accuracy = 1.0 - final_loss  # Simplified accuracy metric
        
        metrics = TrainingMetrics(
            framework=self.config.regulatory_framework,
            model_name=self.config.model_name,
            training_time=training_time,
            final_loss=final_loss,
            final_accuracy=final_accuracy,
            best_epoch=self.trainer.state.best_metric,
            total_steps=train_result.global_step,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            dataset_size=len(train_dataset)
        )
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final loss: {final_loss:.4f}")
        logger.info(f"Final accuracy: {final_accuracy:.4f}")
        
        return metrics
    
    def save_model(self, output_path: str):
        """Save the trained model."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save training config
        config_path = output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Model saved to {output_path}")
    
    def evaluate_model(self, test_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model on test data")
        
        # Tokenize test dataset
        test_tokenized = self.tokenize_data(test_dataset)
        
        # Run evaluation
        eval_results = self.trainer.evaluate(test_tokenized)
        
        logger.info(f"Test evaluation results: {eval_results}")
        
        return eval_results
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response for a given prompt."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip() 