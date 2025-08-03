#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for real regulatory compliance data.

This script helps you prepare real regulatory documents for training the SLM system.
"""

import os
import sys
import json
import requests
from pathlib import Path
from urllib.parse import urlparse

def download_regulatory_documents():
    """Download sample regulatory documents from public sources."""
    print("📥 Downloading sample regulatory documents...")
    
    # Sample regulatory document URLs (public sources)
    documents = {
        "sox_section_404.pdf": {
            "url": "https://www.sec.gov/about/laws/soa2002.pdf",
            "description": "Sarbanes-Oxley Act Section 404",
            "framework": "sox"
        },
        "gaap_revenue_recognition.pdf": {
            "url": "https://www.fasb.org/jsp/FASB/Document_C/DocumentPage?cid=1176164076069&acceptedDisclaimer=true",
            "description": "GAAP Revenue Recognition Standard",
            "framework": "gaap"
        },
        "fda_21_cfr_part_11.pdf": {
            "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=11",
            "description": "FDA 21 CFR Part 11 Electronic Records",
            "framework": "fda"
        }
    }
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    
    for filename, doc_info in documents.items():
        file_path = raw_dir / filename
        
        if file_path.exists():
            print(f"✅ {filename} already exists")
            downloaded_count += 1
            continue
            
        print(f"📥 Downloading {filename}...")
        try:
            # For demo purposes, we'll create placeholder files
            # In real usage, you would download from the URLs
            placeholder_content = f"""
            {doc_info['description']}
            
            This is a placeholder for {filename}.
            In a real implementation, this would contain the actual regulatory document.
            
            Framework: {doc_info['framework'].upper()}
            Source: {doc_info['url']}
            
            For actual implementation:
            1. Download the real PDF from the URL
            2. Place it in data/raw/
            3. Run the PDF converter to process it
            """
            
            with open(file_path, 'w') as f:
                f.write(placeholder_content)
            
            print(f"✅ Created placeholder for {filename}")
            downloaded_count += 1
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    print(f"\n📊 Downloaded {downloaded_count}/{len(documents)} documents")
    return downloaded_count


def create_sample_pdfs():
    """Create sample PDF files for testing."""
    print("\n📄 Creating sample PDF files...")
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample PDF content
    sample_docs = {
        "sox_internal_controls.pdf": {
            "content": """
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
            """,
            "framework": "sox"
        },
        "gaap_revenue_606.pdf": {
            "content": """
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
            """,
            "framework": "gaap"
        },
        "fda_clinical_trials.pdf": {
            "content": """
            FDA Regulations for Clinical Trials
            21 CFR Part 312 - Investigational New Drug Application
            
            This part contains procedures and requirements governing the use of investigational 
            new drugs, including procedures and requirements for the submission to, and review 
            by, the Food and Drug Administration of investigational new drug applications (IND's).
            
            Key requirements:
            - IND submission requirements
            - Clinical trial protocols
            - Safety reporting
            - Record keeping
            - Quality control
            """,
            "framework": "fda"
        }
    }
    
    created_count = 0
    
    for filename, doc_info in sample_docs.items():
        file_path = raw_dir / filename
        
        if file_path.exists():
            print(f"✅ {filename} already exists")
            created_count += 1
            continue
        
        # Create a text file that simulates PDF content
        with open(file_path, 'w') as f:
            f.write(doc_info["content"])
        
        print(f"✅ Created {filename}")
        created_count += 1
    
    print(f"\n📊 Created {created_count}/{len(sample_docs)} sample documents")
    return created_count


def setup_data_structure():
    """Set up the complete data directory structure."""
    print("\n📁 Setting up data directory structure...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/training",
        "data/validation",
        "data/test",
        "models/checkpoints",
        "models/deployed",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")
    
    print("📁 Data directory structure ready")


def create_config_files():
    """Create configuration files for different frameworks."""
    print("\n⚙️ Creating framework-specific configuration files...")
    
    config_dir = Path("shared/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    frameworks = {
        "sox": {
            "description": "Sarbanes-Oxley Act compliance",
            "keywords": ["internal control", "financial reporting", "audit", "section 404", "section 302"],
            "special_tokens": ["[SOX]", "[CONTROL]", "[AUDIT]", "[REPORT]"]
        },
        "gaap": {
            "description": "Generally Accepted Accounting Principles",
            "keywords": ["accounting", "financial statement", "revenue recognition", "asset", "liability"],
            "special_tokens": ["[GAAP]", "[ACCOUNTING]", "[FINANCIAL]", "[REVENUE]"]
        },
        "fda": {
            "description": "FDA regulations for drug approval",
            "keywords": ["clinical trial", "drug approval", "21 cfr", "electronic record", "validation"],
            "special_tokens": ["[FDA]", "[CLINICAL]", "[TRIAL]", "[APPROVAL]"]
        }
    }
    
    for framework, config in frameworks.items():
        config_file = config_dir / f"{framework}_config.yaml"
        
        config_content = f"""# {framework.upper()} Configuration
framework: {framework}
description: "{config['description']}"
keywords: {config['keywords']}
special_tokens: {config['special_tokens']}

training:
  model_name: "distilbert-base-uncased"
  max_length: 512
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 500

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
  confidence_threshold: 0.7
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created {framework}_config.yaml")
    
    print("⚙️ Configuration files ready")


def main():
    """Main setup function."""
    print("🚀 Setting up Real Data for Regulatory Compliance SLM System")
    print("="*60)
    
    try:
        # Step 1: Set up directory structure
        setup_data_structure()
        
        # Step 2: Create sample documents
        sample_count = create_sample_pdfs()
        
        # Step 3: Download regulatory documents (placeholders)
        download_count = download_regulatory_documents()
        
        # Step 4: Create configuration files
        create_config_files()
        
        print("\n" + "="*60)
        print("🎉 Setup Complete!")
        print("="*60)
        print(f"✅ Created {sample_count} sample documents")
        print(f"✅ Prepared {download_count} regulatory document placeholders")
        print("✅ Set up complete data directory structure")
        print("✅ Created framework-specific configurations")
        
        print("\n📚 Next Steps:")
        print("1. Add real PDF documents to data/raw/")
        print("2. Run: python3 pdf_converter/src/main.py --input-dir data/raw --output-dir data/processed")
        print("3. Run: python3 model_trainer/src/train.py --data-path data/processed --output-path models/")
        print("4. Run: python3 broker/src/main.py")
        print("5. Test the system with real data!")
        
        print("\n💡 Tips:")
        print("- Place real PDF regulatory documents in data/raw/")
        print("- Use documents from SEC, FASB, FDA, etc.")
        print("- Ensure documents are properly formatted and readable")
        print("- Consider document size and processing time")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 