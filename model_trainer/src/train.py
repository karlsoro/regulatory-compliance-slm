#!/usr/bin/env python3
"""
Main entry point for the Model Trainer application.

This script provides a command-line interface for training small language models
for regulatory compliance tasks.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional

from trainer import ModelTrainer, TrainingConfig


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('model_trainer.log')
        ]
    )


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return TrainingConfig(**config_data)


def main():
    """Main function for the model trainer CLI."""
    parser = argparse.ArgumentParser(
        description="Train small language models for regulatory compliance"
    )
    
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to processed training data directory"
    )
    
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to save trained model"
    )
    
    parser.add_argument(
        "--config",
        help="Path to training configuration file (JSON)"
    )
    
    parser.add_argument(
        "--framework",
        choices=["sox", "gaap", "fda", "hipaa", "gdpr"],
        help="Regulatory framework to train for (if not in config)"
    )
    
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Base model to use for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be trained without actually training"
    )
    
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate an existing model (requires --model-path)"
    )
    
    parser.add_argument(
        "--model-path",
        help="Path to existing model for evaluation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create config from command line arguments
        config = TrainingConfig(
            model_name=args.model_name,
            regulatory_framework=args.framework or "sox",
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            output_dir=args.output_path
        )
        logger.info("Created configuration from command line arguments")
    
    # Validate paths
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting model training for {config.regulatory_framework}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No training will be performed")
        return
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        if args.evaluate_only:
            if not args.model_path:
                logger.error("--model-path is required for evaluation-only mode")
                sys.exit(1)
            
            # Load existing model and evaluate
            logger.info(f"Loading model from {args.model_path}")
            # Note: This would require implementing model loading functionality
            logger.warning("Model loading not implemented in this version")
            return
        
        # Prepare data
        logger.info("Preparing training data...")
        train_dataset, val_dataset, test_dataset = trainer.prepare_data(str(data_path))
        
        # Train model
        logger.info("Starting model training...")
        metrics = trainer.train(train_dataset, val_dataset)
        
        # Save model
        logger.info("Saving trained model...")
        trainer.save_model(str(output_path))
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics = trainer.evaluate_model(test_dataset)
        
        # Save training results
        results = {
            "training_metrics": asdict(metrics),
            "test_metrics": test_metrics,
            "config": asdict(config)
        }
        
        results_path = output_path / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Training time: {metrics.training_time:.2f} seconds")
        logger.info(f"Final loss: {metrics.final_loss:.4f}")
        logger.info(f"Final accuracy: {metrics.final_accuracy:.4f}")
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"Results saved to: {results_path}")
        
        # Test model with a sample prompt
        logger.info("Testing model with sample prompt...")
        sample_prompt = f"Generate {config.regulatory_framework.upper()}-compliant documentation for financial controls"
        response = trainer.generate_response(sample_prompt)
        logger.info(f"Sample prompt: {sample_prompt}")
        logger.info(f"Model response: {response}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 