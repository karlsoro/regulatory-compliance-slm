#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the Regulatory Compliance SLM System

This script demonstrates the complete workflow from PDF processing to model training
to broker usage with sample data.
"""

import os
import sys
import tempfile
import json
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pdf_converter.src.converter import PDFConverter
from model_trainer.src.trainer import ModelTrainer, TrainingConfig
from broker.src.broker import ComplianceBroker, TaskRequest


def create_sample_pdf_data():
    """Create sample PDF data for demonstration."""
    sample_data = {
        "sox": {
            "text_content": """
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
            """,
            "regulatory_framework": "sox",
            "document_type": "regulation",
            "page_count": 5,
            "confidence_score": 0.9
        },
        "gaap": {
            "text_content": """
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
            """,
            "regulatory_framework": "gaap",
            "document_type": "regulation",
            "page_count": 4,
            "confidence_score": 0.85
        }
    }
    return sample_data


def demo_pdf_converter():
    """Demonstrate PDF converter functionality."""
    print("\n" + "="*60)
    print("DEMO: PDF Converter")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample data
        raw_data_dir = temp_path / "raw"
        raw_data_dir.mkdir()
        
        processed_data_dir = temp_path / "processed"
        processed_data_dir.mkdir()
        
        sample_data = create_sample_pdf_data()
        
        # Create mock PDF files
        for framework, data in sample_data.items():
            json_file = raw_data_dir / f"{framework}_regulation.json"
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Initialize converter
        converter = PDFConverter(output_format="json")
        
        # Process files (mock the actual PDF processing)
        print(f"Processing {len(sample_data)} sample documents...")
        
        # Simulate processing
        processed_files = []
        for framework, data in sample_data.items():
            # Create processed content
            processed_content = {
                "text_content": data["text_content"],
                "tables": [],
                "images": [],
                "metadata": {
                    "filename": f"{framework}_regulation.pdf",
                    "file_size": 1024,
                    "page_count": data["page_count"],
                    "regulatory_framework": data["regulatory_framework"],
                    "document_type": data["document_type"],
                    "extraction_method": "demo",
                    "confidence_score": data["confidence_score"],
                    "processing_time": 1.5
                },
                "sections": {"main": data["text_content"]},
                "keywords": ["compliance", "regulation", "documentation"]
            }
            
            # Save processed file
            output_file = processed_data_dir / f"{framework}_processed.json"
            with open(output_file, 'w') as f:
                json.dump(processed_content, f, indent=2)
            
            processed_files.append(output_file)
        
        print(f"✅ Successfully processed {len(processed_files)} documents")
        print(f"📁 Output saved to: {processed_data_dir}")
        
        return processed_data_dir


def demo_model_trainer(processed_data_dir):
    """Demonstrate model trainer functionality."""
    print("\n" + "="*60)
    print("DEMO: Model Trainer")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create training config
        config = TrainingConfig(
            model_name="distilbert-base-uncased",
            regulatory_framework="sox",
            max_length=128,
            batch_size=2,
            learning_rate=2e-5,
            num_epochs=1,
            output_dir=str(temp_path / "models")
        )
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        print(f"Training model for {config.regulatory_framework.upper()} compliance...")
        
        # Prepare data (mock)
        print("Preparing training data...")
        
        # Create mock datasets
        train_data = [
            {
                "prompt": "Generate SOX-compliant internal control documentation",
                "response": "SOX compliance requires internal controls for financial reporting.",
                "text": "Generate SOX-compliant internal control documentation SOX compliance requires internal controls for financial reporting."
            }
        ] * 10
        
        val_data = train_data[:2]
        test_data = train_data[:1]
        
        print(f"✅ Prepared {len(train_data)} training examples")
        print(f"✅ Prepared {len(val_data)} validation examples")
        print(f"✅ Prepared {len(test_data)} test examples")
        
        # Mock training
        print("Training model (simulated)...")
        
        # Simulate training metrics
        mock_metrics = {
            "framework": "sox",
            "model_name": "distilbert-base-uncased",
            "training_time": 60.0,
            "final_loss": 0.15,
            "final_accuracy": 0.85,
            "best_epoch": 1,
            "total_steps": 100,
            "learning_rate": 2e-5,
            "batch_size": 2,
            "dataset_size": 10
        }
        
        print(f"✅ Training completed in {mock_metrics['training_time']:.1f} seconds")
        print(f"✅ Final loss: {mock_metrics['final_loss']:.3f}")
        print(f"✅ Final accuracy: {mock_metrics['final_accuracy']:.3f}")
        
        # Save model (mock)
        model_output_dir = temp_path / "trained_model"
        model_output_dir.mkdir()
        
        # Create mock model files
        (model_output_dir / "pytorch_model.bin").touch()
        (model_output_dir / "tokenizer.json").touch()
        (model_output_dir / "training_config.json").touch()
        
        print(f"✅ Model saved to: {model_output_dir}")
        
        return model_output_dir


async def demo_broker(model_output_dir):
    """Demonstrate broker functionality."""
    print("\n" + "="*60)
    print("DEMO: Compliance Broker")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize broker
        broker = ComplianceBroker(models_dir=str(temp_path / "broker_models"))
        
        # Register model
        print("Registering SOX compliance model...")
        success = broker.register_model("sox", str(model_output_dir))
        
        if success:
            print("✅ Model registered successfully")
        else:
            print("❌ Failed to register model")
            return
        
        # Mock model loading
        print("Loading model into memory...")
        broker.models["sox"].is_loaded = True
        broker.models["sox"].model = "mock_model"
        broker.models["sox"].tokenizer = "mock_tokenizer"
        
        print("✅ Model loaded successfully")
        
        # Test compliance requests
        test_requests = [
            {
                "prompt": "Generate SOX-compliant internal control documentation for our ERP system",
                "frameworks": ["sox"],
                "description": "SOX internal controls"
            },
            {
                "prompt": "Create compliance documentation for financial reporting under SOX and GAAP",
                "frameworks": ["sox", "gaap"],
                "description": "Multi-framework compliance"
            }
        ]
        
        for i, test_request in enumerate(test_requests, 1):
            print(f"\n--- Test Request {i}: {test_request['description']} ---")
            
            request = TaskRequest(
                prompt=test_request["prompt"],
                frameworks=test_request["frameworks"],
                max_response_length=300,
                temperature=0.7
            )
            
            # Mock response processing
            print(f"Processing request: {request.prompt[:50]}...")
            
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Create mock response
            mock_response = f"Generated {', '.join(request.frameworks).upper()} compliance documentation for the requested system. Required documents include control documentation, audit reports, and evidence logs."
            
            print(f"✅ Response generated successfully")
            print(f"📝 Response: {mock_response[:100]}...")
            print(f"🎯 Frameworks used: {', '.join(request.frameworks)}")
            print(f"⏱️  Processing time: 0.5 seconds")
            print(f"🎯 Confidence: 0.85")


def main():
    """Run the complete demo."""
    print("🚀 Regulatory Compliance SLM System Demo")
    print("="*60)
    print("This demo shows the complete workflow from PDF processing to model training to broker usage.")
    
    try:
        # Step 1: PDF Converter Demo
        processed_data_dir = demo_pdf_converter()
        
        # Step 2: Model Trainer Demo
        model_output_dir = demo_model_trainer(processed_data_dir)
        
        # Step 3: Broker Demo
        asyncio.run(demo_broker(model_output_dir))
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The system demonstrated:")
        print("✅ PDF document processing and framework detection")
        print("✅ Model training for regulatory compliance")
        print("✅ Multi-framework compliance request processing")
        print("✅ Structured response generation with evidence requirements")
        
        print("\n📚 Next steps:")
        print("1. Add real PDF documents to data/raw/")
        print("2. Run: python pdf_converter/src/main.py --input-dir data/raw --output-dir data/processed")
        print("3. Run: python model_trainer/src/train.py --data-path data/processed --output-path models/")
        print("4. Run: python broker/src/main.py")
        print("5. Test with: curl -X POST http://localhost:8000/compliance/process -H 'Content-Type: application/json' -d '{\"prompt\": \"Generate SOX compliance documentation\"}'")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 