# Quick Reference Guide - Regulatory Compliance SLM

## System Components Overview

### 📁 Project Structure
```
regulatory-compliance-slm/
├── pdf_converter/          # PDF processing and text extraction
├── model_trainer/          # Model training and fine-tuning
├── broker/                 # Multi-model orchestration
├── data/                   # Raw and processed data
├── models/                 # Trained model checkpoints
├── scripts/                # Utility and test scripts
├── tests/                  # Unit and integration tests
└── shared/                 # Shared configuration
```

### 🔧 Key Components

#### 1. PDF Converter (`pdf_converter/`)
**Purpose:** Extract and structure text from PDF documents
- **Input:** PDF files
- **Output:** Structured JSON with metadata
- **Key Files:** `converter.py`, `main.py`

#### 2. Model Trainer (`model_trainer/`)
**Purpose:** Train specialized models for each regulatory framework
- **Input:** Structured JSON data
- **Output:** Trained model checkpoints
- **Key Files:** `trainer.py`, `train.py`

#### 3. Compliance Broker (`broker/`)
**Purpose:** Route queries to appropriate specialized models
- **Input:** Compliance queries
- **Output:** Aggregated responses
- **Key Files:** `broker.py`, `main.py`

## 🚀 Quick Start Commands

### Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Process Documents
```bash
# Convert PDFs to structured data
python3 pdf_converter/src/main.py --input-dir data/raw --output-dir data/processed
```

### Train Model
```bash
# Train SOX compliance model
python3 model_trainer/src/train.py \
    --data-path data/processed \
    --output-path models/checkpoints \
    --framework sox \
    --epochs 3 \
    --batch-size 4
```

### Test Model
```bash
# Test trained model
python3 scripts/test_sox_advanced.py
```

### Run Broker
```bash
# Start broker API server
python3 broker/src/main.py
```

## 🔍 Key Concepts Explained

### Small Language Models (SLMs)
- **What:** Specialized models trained for specific tasks
- **Why:** Better performance than general models for domain-specific tasks
- **How:** Fine-tune pre-trained models on specialized data

### Regulatory Frameworks
- **SOX:** Sarbanes-Oxley Act (financial reporting)
- **GAAP:** Generally Accepted Accounting Principles
- **FDA:** Food and Drug Administration regulations
- **HIPAA:** Health Insurance Portability and Accountability Act
- **GDPR:** General Data Protection Regulation

### Model Architecture
- **Base Model:** GPT-2 (causal language model)
- **Fine-tuning:** Adapt pre-trained model to regulatory domain
- **Specialization:** One model per regulatory framework

### Broker Pattern
- **Purpose:** Route queries to appropriate specialized models
- **Benefits:** Handle multi-framework queries efficiently
- **Implementation:** Async processing with response aggregation

## 🛠️ Common Tasks

### Adding New Regulatory Framework
1. **Add Framework Data:**
   ```bash
   mkdir -p data/raw/new_framework
   # Add PDF documents
   ```

2. **Update Configuration:**
   ```yaml
   # In shared/config/training_config.yaml
   frameworks:
     new_framework:
       description: "New Framework Description"
       keywords: ["keyword1", "keyword2"]
   ```

3. **Train Model:**
   ```bash
   python3 model_trainer/src/train.py \
       --framework new_framework \
       --data-path data/processed \
       --output-path models/checkpoints/new_framework
   ```

### Testing System Components
```bash
# Test structure
python3 scripts/test_structure.py

# Test components
python3 scripts/test_components.py

# Test end-to-end
python3 scripts/demo.py
```

### Monitoring Training
```bash
# Check training logs
tail -f model_trainer.log

# Monitor GPU usage (if available)
nvidia-smi

# Check model checkpoints
ls -la models/checkpoints/
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError"
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### 2. "CUDA out of memory"
```bash
# Solution: Reduce batch size
python3 model_trainer/src/train.py --batch-size 1
```

#### 3. "Model path does not exist"
```bash
# Solution: Check if training completed
ls -la models/checkpoints/
# Re-run training if needed
```

#### 4. "PDF extraction failed"
```python
# Solution: Check PDF format
# Some PDFs may be image-based or corrupted
# Try different extraction methods
```

#### 5. "Training stuck or slow"
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Monitor system resources
htop  # or top
```

### Performance Optimization

#### Training Speed
- **Use GPU:** Ensure CUDA is available
- **Increase Batch Size:** If memory allows
- **Mixed Precision:** Use float16 for GPU training

#### Inference Speed
- **Model Quantization:** Reduce precision
- **Batch Processing:** Process multiple queries together
- **Caching:** Store frequent responses

#### Memory Usage
- **Reduce Batch Size:** For limited memory
- **Gradient Accumulation:** Simulate larger batches
- **Model Sharding:** Split large models

## 📊 Evaluation Metrics

### Training Metrics
- **Loss:** Training and validation loss
- **Perplexity:** Language model quality measure
- **Accuracy:** For classification tasks

### Model Quality
- **Relevance:** Response relevance to query
- **Completeness:** Coverage of compliance requirements
- **Accuracy:** Factual correctness

### System Performance
- **Response Time:** Query processing speed
- **Throughput:** Queries per second
- **Resource Usage:** CPU, memory, GPU utilization

## 🔒 Security Considerations

### Data Privacy
- **Local Processing:** No data sent to external APIs
- **Data Retention:** Configurable retention policies
- **Access Control:** Restrict model access

### Model Security
- **Input Validation:** Sanitize user inputs
- **Rate Limiting:** Prevent abuse
- **Audit Logging:** Track usage and access

### Compliance
- **Documentation:** Maintain compliance records
- **Validation:** Regular model validation
- **Updates:** Keep models current with regulations

## 📈 Best Practices

### Development
1. **Start Small:** Begin with one framework
2. **Test Incrementally:** Validate each component
3. **Version Control:** Track all changes
4. **Documentation:** Maintain clear documentation

### Training
1. **Data Quality:** Ensure clean, relevant training data
2. **Validation:** Use separate validation sets
3. **Monitoring:** Track training progress
4. **Checkpointing:** Save progress regularly

### Deployment
1. **Testing:** Comprehensive testing before deployment
2. **Monitoring:** Track system performance
3. **Backup:** Regular model and data backups
4. **Updates:** Plan for model updates and maintenance

### Maintenance
1. **Regular Evaluation:** Assess model performance
2. **Data Updates:** Keep training data current
3. **Model Updates:** Retrain with new data
4. **System Monitoring:** Track system health

## 🎯 Quick Tips

### Data Preparation
- **Clean Data:** Remove irrelevant content
- **Balance Datasets:** Equal representation across frameworks
- **Validate Format:** Ensure consistent JSON structure

### Model Training
- **Start with Few Epochs:** 1-3 epochs for initial testing
- **Monitor Loss:** Stop if loss increases
- **Save Checkpoints:** Enable training resumption

### System Integration
- **Test Components:** Validate each component individually
- **Handle Errors:** Implement graceful error handling
- **Log Everything:** Comprehensive logging for debugging

### Performance
- **Profile Code:** Identify bottlenecks
- **Optimize Critical Paths:** Focus on performance-critical areas
- **Scale Gradually:** Start small and scale as needed

This quick reference provides essential information for working with the regulatory compliance SLM system. Refer to the main documentation for detailed explanations and examples.
