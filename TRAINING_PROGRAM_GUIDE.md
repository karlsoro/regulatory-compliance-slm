# Regulatory Compliance SLM Training Program - Complete Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Step-by-Step Build Guide](#step-by-step-build-guide)
4. [Key Implementation Details](#key-implementation-details)
5. [Testing Strategy](#testing-strategy)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Best Practices](#best-practices)

## System Overview

This project demonstrates how to build a **Small Language Model (SLM)** training system for regulatory compliance. The system processes regulatory documents, trains specialized models, and provides compliance guidance through an intelligent broker system.

### Why This Approach?

**Traditional Compliance Challenges:**
- Manual document review is time-consuming and error-prone
- General AI models lack domain-specific knowledge
- Complex compliance often spans multiple regulatory frameworks

**Our Solution:**
- **Specialized Models:** One model per regulatory framework (SOX, GAAP, FDA, etc.)
- **Intelligent Broker:** Routes queries to appropriate specialized models
- **Structured Processing:** Converts unstructured PDFs into training data
- **Scalable Architecture:** Easy to add new frameworks and data sources

## Component Architecture

### 1. PDF Converter (`pdf_converter/`)

**Purpose:** Transforms PDF documents into structured training data.

**Key Components:**
- `converter.py`: Core extraction and classification logic
- `main.py`: CLI interface for batch processing

**What It Does:**
```python
# Extracts text using multiple methods for robustness
extractors = [
    PyPDF2Extractor(),      # Basic text extraction
    PdfPlumberExtractor(),  # Better table handling
    PyMuPDFExtractor()      # Advanced formatting preservation
]

# Classifies documents by regulatory framework
def classify_document(self, content: str) -> str:
    # Uses keyword matching and ML classification
    # Returns: 'sox', 'gaap', 'fda', etc.
```

**Why Multiple Extractors?**
- PDFs vary in quality and format
- Some extractors work better with tables, others with text
- Redundancy ensures we capture content even from problematic PDFs

### 2. Model Trainer (`model_trainer/`)

**Purpose:** Trains specialized language models for each regulatory framework.

**Key Components:**
- `trainer.py`: Core training logic with fine-tuning
- `train.py`: CLI interface for model training

**What It Does:**
```python
# Filters data by regulatory framework
def _filter_by_framework(self, data):
    return [item for item in data 
            if item.get('regulatory_framework', '').lower() == 
            self.config.regulatory_framework.lower()]

# Fine-tunes pre-trained models
def setup_model(self):
    # Uses GPT-2 for text generation tasks
    # DistilBERT is for classification, not generation
    model_name = "gpt2" if "distilbert" in self.config.model_name.lower() else self.config.model_name
```

**Why GPT-2 for Fine-tuning?**
- **Causal Language Model:** Generates text sequentially (good for compliance guidance)
- **Smaller Size:** Faster training and inference than larger models
- **Good Base:** Pre-trained on diverse text, adapts well to specialized domains

### 3. Compliance Broker (`broker/`)

**Purpose:** Orchestrates multiple specialized models for complex compliance queries.

**Key Components:**
- `broker.py`: Core orchestration logic
- `main.py`: FastAPI server for API endpoints

**What It Does:**
```python
# Routes queries to appropriate models
async def process_query(self, query: str) -> ComplianceResponse:
    # Analyzes query to determine relevant frameworks
    # Routes to specialized models
    # Aggregates responses into coherent answer
```

**Why a Broker?**
- **Multi-Framework Queries:** "What do I need for SOX and FDA compliance?"
- **Specialized Responses:** Each model knows its domain best
- **Unified Interface:** Single API for all compliance needs

## Step-by-Step Build Guide

### Phase 1: Environment Setup

```bash
# 1. Create project structure
mkdir regulatory-compliance-slm
cd regulatory-compliance-slm

# 2. Set up Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Key Dependencies Explained:**
- `transformers`: Pre-trained models and training utilities
- `torch`: Deep learning framework
- `fastapi`: API framework for broker
- `pdfplumber`: PDF text extraction
- `datasets`: Data loading and processing

### Phase 2: Data Preparation

```bash
# 1. Create sample data structure
python3 scripts/setup_real_data.py

# 2. Process text documents (simulating PDFs)
python3 scripts/process_text_documents.py
```

**What This Does:**
- Creates directory structure for different regulatory frameworks
- Generates sample text files simulating PDF content
- Processes text into structured JSON format for training

### Phase 3: Model Training

```bash
# Train SOX compliance model
python3 model_trainer/src/train.py \
    --data-path data/processed \
    --output-path models/checkpoints \
    --framework sox \
    --epochs 1 \
    --batch-size 2
```

**Training Parameters Explained:**
- `--epochs 1`: Number of complete passes through training data
- `--batch-size 2`: Number of samples processed together (smaller = less memory)
- `--framework sox`: Filters data for SOX-specific training

### Phase 4: Model Testing

```bash
# Test the trained model
python3 scripts/test_sox_advanced.py
```

**What to Look For:**
- Model loads without errors
- Generates relevant SOX compliance content
- Responses are coherent and domain-appropriate

## Key Implementation Details

### 1. Data Processing Pipeline

**Text Extraction Strategy:**
```python
def extract_text(self, pdf_path: str) -> str:
    # Try multiple extractors in order of preference
    for extractor in self.extractors:
        try:
            text = extractor.extract(pdf_path)
            if text.strip():  # Check if we got meaningful content
                return text
        except Exception as e:
            logger.warning(f"Extractor {extractor.__class__.__name__} failed: {e}")
    
    raise ValueError("All extractors failed")
```

**Why This Approach?**
- **Robustness:** If one method fails, others may succeed
- **Quality:** Different extractors handle different PDF types better
- **Logging:** Tracks which methods work for optimization

### 2. Model Training Configuration

**Training Arguments:**
```python
training_args = TrainingArguments(
    output_dir=self.config.output_dir,
    num_train_epochs=self.config.epochs,
    per_device_train_batch_size=self.config.batch_size,
    save_steps=500,  # Save checkpoint every 500 steps
    logging_steps=100,  # Log progress every 100 steps
    learning_rate=2e-5,  # Small learning rate for fine-tuning
    warmup_steps=100,  # Gradual warmup to prevent instability
)
```

**Why These Settings?**
- **Small Learning Rate:** Prevents overwriting pre-trained knowledge
- **Warmup Steps:** Gradually increases learning rate to prevent early instability
- **Frequent Logging:** Monitor training progress and catch issues early

### 3. Broker Query Processing

**Query Analysis:**
```python
def analyze_query(self, query: str) -> List[str]:
    # Extract regulatory frameworks mentioned in query
    frameworks = []
    query_lower = query.lower()
    
    if 'sox' in query_lower or 'sarbanes' in query_lower:
        frameworks.append('sox')
    if 'gaap' in query_lower or 'accounting' in query_lower:
        frameworks.append('gaap')
    # Add more framework detection logic
    
    return frameworks
```

**Response Aggregation:**
```python
def aggregate_responses(self, responses: List[str]) -> str:
    # Combine responses from multiple models
    combined = "\n\n".join([
        f"## {framework.upper()} Compliance\n{response}"
        for framework, response in responses
    ])
    return combined
```

## Testing Strategy

### 1. Unit Testing
```bash
# Test individual components
python3 scripts/test_components.py
```

**What Gets Tested:**
- PDF converter functionality
- Model trainer data processing
- Broker query routing
- Import statements and basic functionality

### 2. Integration Testing
```bash
# Test complete workflow
python3 scripts/demo.py
```

**What Gets Tested:**
- End-to-end data processing
- Model training pipeline
- Broker integration
- API endpoints

### 3. Model Validation
```bash
# Test trained model performance
python3 scripts/test_sox_advanced.py
```

**What Gets Tested:**
- Model loading and inference
- Response quality and relevance
- Error handling and edge cases

## Troubleshooting Guide

### Common Issues and Solutions

**1. "ModuleNotFoundError: No module named 'transformers'"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**2. "CUDA out of memory" during training**
```bash
# Solution: Reduce batch size
python3 model_trainer/src/train.py --batch-size 1
```

**3. "Model path does not exist"**
```bash
# Solution: Check model directory
ls -la models/checkpoints/
# Ensure training completed successfully
```

**4. "PDF extraction failed"**
```python
# Solution: Check PDF format and try different extractors
# Some PDFs may be image-based or corrupted
```

### Performance Optimization

**Training Speed:**
- Use GPU if available (`torch.cuda.is_available()`)
- Increase batch size (if memory allows)
- Use mixed precision training

**Inference Speed:**
- Model quantization (reduce precision)
- Batch processing for multiple queries
- Caching frequently requested responses

## Best Practices

### 1. Data Quality
- **Validate Input:** Check PDF quality before processing
- **Clean Data:** Remove irrelevant content and formatting
- **Balance Datasets:** Ensure equal representation across frameworks

### 2. Model Training
- **Start Small:** Begin with few epochs and small datasets
- **Monitor Metrics:** Track loss, accuracy, and perplexity
- **Save Checkpoints:** Enable training resumption if interrupted

### 3. System Design
- **Modular Architecture:** Keep components independent
- **Error Handling:** Graceful degradation when components fail
- **Logging:** Comprehensive logging for debugging and monitoring

### 4. Deployment
- **Containerization:** Use Docker for consistent environments
- **API Design:** RESTful endpoints with proper error responses
- **Monitoring:** Track system performance and model quality

## Key Tips and Hints

### 1. Data Preparation
- **Start with Text Files:** Easier to debug than PDF processing
- **Use Sample Data:** Validate pipeline before processing real documents
- **Structure Matters:** Consistent JSON format is crucial for training

### 2. Model Selection
- **GPT-2 for Generation:** Use for compliance guidance and document generation
- **DistilBERT for Classification:** Use for document categorization
- **Size vs. Speed:** Balance model size with inference speed requirements

### 3. Training Strategy
- **Transfer Learning:** Always start with pre-trained models
- **Domain Adaptation:** Fine-tune on regulatory-specific data
- **Evaluation:** Use domain-specific metrics, not just general accuracy

### 4. System Integration
- **Async Processing:** Use for handling multiple concurrent requests
- **Caching:** Store frequently accessed model outputs
- **Fallbacks:** Plan for when models or services are unavailable

## Building Your Own System

### Step 1: Define Requirements
- What regulatory frameworks do you need?
- What types of queries will you handle?
- What performance requirements do you have?

### Step 2: Data Collection
- Gather regulatory documents for each framework
- Ensure data quality and consistency
- Create structured training datasets

### Step 3: Model Development
- Start with one framework (e.g., SOX)
- Validate the pipeline end-to-end
- Expand to additional frameworks

### Step 4: System Integration
- Implement the broker for multi-framework queries
- Add API endpoints for external access
- Deploy with monitoring and logging

### Step 5: Validation and Testing
- Test with real-world queries
- Validate compliance accuracy
- Monitor system performance

This guide provides the foundation for building effective regulatory compliance AI systems. The modular architecture allows for easy expansion and customization based on specific requirements.
