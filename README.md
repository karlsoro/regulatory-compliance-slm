# Regulatory Compliance Small Language Model (SLM) System

A comprehensive system for training and deploying small language models specifically designed for regulatory compliance in highly regulated industries. This project addresses the complex challenge of automating compliance documentation generation for frameworks like SOX, GAAP, FDA, and HIPAA.

## Project Overview

This system consists of two main applications:

1. **PDF Converter** - Processes regulatory documents and company data for training
2. **Model Trainer** - Builds and fine-tunes specialized SLMs for different regulatory frameworks
3. **Broker** - Orchestrates multiple SLMs to handle complex compliance requests

## Architecture

```
regulatory-compliance-slm/
├── pdf_converter/          # PDF processing application
│   ├── src/               # Source code for PDF conversion
│   └── tests/             # PDF converter tests
├── model_trainer/         # SLM training application
│   ├── src/               # Training pipeline code
│   └── tests/             # Model training tests
├── broker/                # Model orchestration service
│   ├── src/               # Broker API and routing logic
│   └── tests/             # Broker tests
├── shared/                # Shared utilities and configuration
│   ├── utils/             # Common utilities
│   └── config/            # Configuration files
├── data/                  # Data storage
│   ├── raw/               # Original PDFs and documents
│   ├── processed/         # Cleaned and structured data
│   └── training/          # Training datasets
├── models/                # Trained models
│   ├── checkpoints/       # Model checkpoints during training
│   └── deployed/          # Production-ready models
├── tests/                 # End-to-end testing
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL (for data storage)
- Redis (for caching)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd regulatory-compliance-slm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
python scripts/setup_database.py
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Applications

#### PDF Converter
```bash
cd pdf_converter
python src/main.py --input-dir ../data/raw --output-dir ../data/processed
```

#### Model Trainer
```bash
cd model_trainer
python src/train.py --config ../shared/config/training_config.yaml
```

#### Broker Service
```bash
cd broker
python src/main.py
```

## Testing

Run the complete test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v
```

## Usage Examples

### Converting PDF Documents
```python
from pdf_converter.src.converter import PDFConverter

converter = PDFConverter()
converter.process_folder("data/raw/sox_documents", "data/processed/sox")
```

### Training a SOX Compliance Model
```python
from model_trainer.src.trainer import ModelTrainer

trainer = ModelTrainer("sox")
trainer.train(
    data_path="data/training/sox_dataset.json",
    output_path="models/checkpoints/sox_model"
)
```

### Using the Broker for Compliance Requests
```python
from broker.src.client import BrokerClient

client = BrokerClient()
response = client.process_request(
    "Generate SOX-compliant internal control documentation for our ERP system"
)
print(response)
```

## Supported Regulatory Frameworks

- **SOX (Sarbanes-Oxley Act)** - Financial reporting and internal controls
- **GAAP (Generally Accepted Accounting Principles)** - Accounting standards
- **FDA 21 CFR Part 11** - Electronic records and signatures
- **HIPAA** - Healthcare data privacy and security
- **GDPR** - Data protection and privacy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please open an issue in the repository. 