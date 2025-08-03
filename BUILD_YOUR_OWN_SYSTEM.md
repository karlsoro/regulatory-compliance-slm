# How to Build Your Own Regulatory Compliance SLM System

## Prerequisites

Before starting, ensure you have:
- Python 3.8+ installed
- Git for version control
- Basic understanding of machine learning concepts
- Familiarity with regulatory compliance (SOX, GAAP, FDA, etc.)

## Step-by-Step Implementation Guide

### Step 1: Project Setup

```bash
# Create project directory
mkdir my-compliance-slm
cd my-compliance-slm

# Initialize git repository
git init

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create project structure
mkdir -p {pdf_converter,model_trainer,broker}/{src,tests}
mkdir -p {data/{raw,processed},models/{checkpoints,deployed},scripts,shared/config}
mkdir -p tests/{unit,integration,e2e}
```

### Step 2: Dependencies Setup

Create `requirements.txt`:
```txt
# Core ML libraries
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.20.0

# PDF processing
PyPDF2>=3.0.0
pdfplumber>=0.7.0
PyMuPDF>=1.20.0

# Data processing
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# API framework
fastapi>=0.95.0
uvicorn>=0.20.0
pydantic>=1.10.0

# Utilities
python-dotenv>=0.19.0
pyyaml>=6.0
click>=8.1.0
tqdm>=4.64.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: PDF Converter Implementation

Create `pdf_converter/src/converter.py`:
```python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json

@dataclass
class DocumentMetadata:
    filename: str
    regulatory_framework: str
    document_type: str
    extraction_method: str

@dataclass
class ExtractedContent:
    text: str
    metadata: DocumentMetadata
    sections: Dict[str, str]

class PDFConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods."""
        # Implementation with fallback extractors
        pass
        
    def classify_document(self, content: str) -> str:
        """Classify document by regulatory framework."""
        # Keyword-based classification
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['sox', 'sarbanes', 'financial']):
            return 'sox'
        elif any(word in content_lower for word in ['gaap', 'accounting', 'financial']):
            return 'gaap'
        elif any(word in content_lower for word in ['fda', 'medical', 'drug']):
            return 'fda'
        else:
            return 'general'
            
    def process_pdf(self, pdf_path: str) -> ExtractedContent:
        """Main processing function."""
        # Extract text
        text = self.extract_text(pdf_path)
        
        # Classify document
        framework = self.classify_document(text)
        
        # Create metadata
        metadata = DocumentMetadata(
            filename=Path(pdf_path).name,
            regulatory_framework=framework,
            document_type='regulation',
            extraction_method='pdfplumber'
        )
        
        return ExtractedContent(
            text=text,
            metadata=metadata,
            sections={}
        )
```

### Step 4: Model Trainer Implementation

Create `model_trainer/src/trainer.py`:
```python
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer
)
from datasets import Dataset
import json
from pathlib import Path

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_data(self, data_path: str) -> Dataset:
        """Load and preprocess training data."""
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Filter by framework
        framework_data = [
            item for item in data 
            if item.get('regulatory_framework', '').lower() == 
            self.config.regulatory_framework.lower()
        ]
        
        return Dataset.from_list(framework_data)
        
    def setup_model(self):
        """Initialize model and tokenizer."""
        # Use GPT-2 for text generation
        model_name = "gpt2"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        
    def train(self, train_dataset, val_dataset=None):
        """Train the model."""
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=False,
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=None,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized if val_dataset else None,
        )
        
        # Train model
        self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
```

### Step 5: Broker Implementation

Create `broker/src/broker.py`:
```python
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@dataclass
class ModelInfo:
    name: str
    framework: str
    model_path: str
    tokenizer_path: str
    description: str
    keywords: List[str]

@dataclass
class ComplianceResponse:
    query: str
    frameworks: List[str]
    responses: Dict[str, str]
    combined_response: str

class ComplianceBroker:
    def __init__(self):
        self.registered_models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        
    def register_model(self, model_info: ModelInfo):
        """Register a trained model."""
        self.registered_models[model_info.framework] = model_info
        
    def load_model(self, framework: str):
        """Load a model into memory."""
        if framework not in self.registered_models:
            raise ValueError(f"Model for framework {framework} not registered")
            
        model_info = self.registered_models[framework]
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_info.model_path)
        
        self.loaded_models[framework] = {
            'tokenizer': tokenizer,
            'model': model
        }
        
    async def process_query(self, query: str) -> ComplianceResponse:
        """Process a compliance query."""
        # Analyze query to determine relevant frameworks
        frameworks = self._analyze_query(query)
        
        responses = {}
        
        # Process with each relevant model
        for framework in frameworks:
            if framework in self.loaded_models:
                response = await self._generate_response(query, framework)
                responses[framework] = response
                
        # Combine responses
        combined = self._combine_responses(responses)
        
        return ComplianceResponse(
            query=query,
            frameworks=frameworks,
            responses=responses,
            combined_response=combined
        )
        
    def _analyze_query(self, query: str) -> List[str]:
        """Determine which frameworks are relevant to the query."""
        query_lower = query.lower()
        frameworks = []
        
        if any(word in query_lower for word in ['sox', 'sarbanes']):
            frameworks.append('sox')
        if any(word in query_lower for word in ['gaap', 'accounting']):
            frameworks.append('gaap')
        if any(word in query_lower for word in ['fda', 'medical']):
            frameworks.append('fda')
            
        return frameworks
        
    async def _generate_response(self, query: str, framework: str) -> str:
        """Generate response using a specific model."""
        model_data = self.loaded_models[framework]
        tokenizer = model_data['tokenizer']
        model = model_data['model']
        
        # Tokenize input
        inputs = tokenizer(query, return_tensors='pt')
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    def _combine_responses(self, responses: Dict[str, str]) -> str:
        """Combine responses from multiple models."""
        if not responses:
            return "No relevant compliance information found."
            
        combined = []
        for framework, response in responses.items():
            combined.append(f"## {framework.upper()} Compliance\n{response}")
            
        return "\n\n".join(combined)
```

### Step 6: Data Processing Script

Create `scripts/process_data.py`:
```python
import json
from pathlib import Path
from pdf_converter.src.converter import PDFConverter

def process_documents(input_dir: str, output_file: str):
    """Process all PDFs in a directory."""
    converter = PDFConverter()
    processed_data = []
    
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            # Process PDF
            content = converter.process_pdf(str(pdf_file))
            
            # Convert to training format
            training_item = {
                'text': content.text,
                'metadata': {
                    'filename': content.metadata.filename,
                    'regulatory_framework': content.metadata.regulatory_framework,
                    'document_type': content.metadata.document_type
                }
            }
            
            processed_data.append(training_item)
            print(f"Processed: {pdf_file.name}")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            
    # Save processed data
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
        
    print(f"Processed {len(processed_data)} documents")

if __name__ == "__main__":
    process_documents("data/raw", "data/processed/training_data.json")
```

### Step 7: Training Script

Create `scripts/train_model.py`:
```python
import argparse
from pathlib import Path
from model_trainer.src.trainer import ModelTrainer
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    data_path: str
    output_dir: str
    regulatory_framework: str
    epochs: int
    batch_size: int
    learning_rate: float = 2e-5

def train_model(config: TrainingConfig):
    """Train a model for a specific regulatory framework."""
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Load data
    print(f"Loading data for {config.regulatory_framework}...")
    dataset = trainer.load_data(config.data_path)
    print(f"Loaded {len(dataset)} training examples")
    
    # Setup model
    print("Setting up model...")
    trainer.setup_model()
    
    # Train model
    print("Starting training...")
    trainer.train(dataset)
    
    print(f"Training complete! Model saved to {config.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        regulatory_framework=args.framework,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    train_model(config)
```

### Step 8: Testing Implementation

Create `scripts/test_system.py`:
```python
import asyncio
from broker.src.broker import ComplianceBroker, ModelInfo
from pathlib import Path

async def test_system():
    """Test the complete system."""
    # Initialize broker
    broker = ComplianceBroker()
    
    # Register models (assuming they exist)
    sox_model = ModelInfo(
        name="SOX Compliance Model",
        framework="sox",
        model_path="models/checkpoints/sox",
        tokenizer_path="models/checkpoints/sox",
        description="Model trained on SOX compliance documents",
        keywords=["financial", "reporting", "controls"]
    )
    
    broker.register_model(sox_model)
    
    # Load model
    print("Loading SOX model...")
    broker.load_model("sox")
    
    # Test queries
    test_queries = [
        "What are the SOX compliance requirements for financial reporting?",
        "How do I implement internal controls for ERP systems?",
        "What documentation is needed for SOX compliance?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = await broker.process_query(query)
        print(f"Response: {response.combined_response[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_system())
```

### Step 9: Configuration Management

Create `shared/config/training_config.yaml`:
```yaml
# Base model configuration
base_model: "gpt2"
max_length: 512
learning_rate: 2e-5

# Training parameters
epochs: 3
batch_size: 4
warmup_steps: 100
save_steps: 500
logging_steps: 100

# Framework-specific settings
frameworks:
  sox:
    description: "Sarbanes-Oxley Act compliance"
    keywords: ["financial", "reporting", "controls", "audit"]
    training_data_path: "data/processed/sox_data.json"
    
  gaap:
    description: "Generally Accepted Accounting Principles"
    keywords: ["accounting", "financial", "standards", "principles"]
    training_data_path: "data/processed/gaap_data.json"
    
  fda:
    description: "Food and Drug Administration regulations"
    keywords: ["medical", "drug", "safety", "approval"]
    training_data_path: "data/processed/fda_data.json"

# Output configuration
output_dir: "models/checkpoints"
save_total_limit: 2
```

### Step 10: Running the System

```bash
# 1. Process your documents
python3 scripts/process_data.py

# 2. Train a model
python3 scripts/train_model.py \
    --data-path data/processed/training_data.json \
    --output-dir models/checkpoints/sox \
    --framework sox \
    --epochs 3 \
    --batch-size 4

# 3. Test the system
python3 scripts/test_system.py
```

## Key Implementation Tips

### 1. Start Small
- Begin with one regulatory framework
- Use small datasets for initial testing
- Validate each component individually

### 2. Data Quality Matters
- Clean and validate your training data
- Ensure proper document classification
- Balance datasets across frameworks

### 3. Model Selection
- Use GPT-2 for text generation tasks
- Consider model size vs. performance trade-offs
- Start with pre-trained models

### 4. Testing Strategy
- Test components in isolation
- Validate end-to-end workflows
- Monitor model performance metrics

### 5. Production Considerations
- Implement proper error handling
- Add logging and monitoring
- Consider deployment strategies (Docker, cloud)

## Common Pitfalls to Avoid

### 1. Data Issues
- **Problem:** Poor quality PDFs or text extraction
- **Solution:** Use multiple extraction methods and validate output

### 2. Model Training
- **Problem:** Overfitting or poor generalization
- **Solution:** Use validation sets and early stopping

### 3. System Integration
- **Problem:** Components not working together
- **Solution:** Test integration points and handle errors gracefully

### 4. Performance
- **Problem:** Slow inference or training
- **Solution:** Optimize batch sizes and use appropriate hardware

This guide provides a complete foundation for building your own regulatory compliance SLM system. Adapt and extend based on your specific requirements and constraints.
