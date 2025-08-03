#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Document Processor for Regulatory Compliance SLM System

This script processes text documents (our sample files) and converts them to the format
needed for training the SLM system.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pdf_converter.src.converter import DocumentMetadata, ExtractedContent

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def detect_regulatory_framework(text_content: str) -> str:
    """Detect regulatory framework from text content."""
    text_lower = text_content.lower()
    
    if any(keyword in text_lower for keyword in ['sarbanes-oxley', 'sox', 'section 404', 'internal control']):
        return 'sox'
    elif any(keyword in text_lower for keyword in ['gaap', 'accounting', 'revenue recognition', 'financial statement']):
        return 'gaap'
    elif any(keyword in text_lower for keyword in ['fda', 'clinical trial', '21 cfr', 'drug approval']):
        return 'fda'
    elif any(keyword in text_lower for keyword in ['hipaa', 'privacy', 'patient', 'phi']):
        return 'hipaa'
    elif any(keyword in text_lower for keyword in ['gdpr', 'data protection', 'consent']):
        return 'gdpr'
    else:
        return 'unknown'

def classify_document_type(text_content: str) -> str:
    """Classify document type based on content."""
    text_lower = text_content.lower()
    
    if any(keyword in text_lower for keyword in ['regulation', 'act', 'law', 'statute']):
        return 'regulation'
    elif any(keyword in text_lower for keyword in ['guideline', 'guidance', 'standard']):
        return 'guideline'
    elif any(keyword in text_lower for keyword in ['policy', 'procedure', 'manual']):
        return 'policy'
    elif any(keyword in text_lower for keyword in ['report', 'assessment', 'audit']):
        return 'report'
    else:
        return 'document'

def extract_keywords(text_content: str, framework: str) -> List[str]:
    """Extract relevant keywords from text content."""
    text_lower = text_content.lower()
    keywords = []
    
    # Framework-specific keywords
    framework_keywords = {
        'sox': ['internal control', 'financial reporting', 'audit', 'section 404', 'section 302', 'sarbanes-oxley'],
        'gaap': ['accounting', 'financial statement', 'revenue recognition', 'asset', 'liability', 'equity'],
        'fda': ['clinical trial', 'drug approval', '21 cfr', 'electronic record', 'validation', 'ind'],
        'hipaa': ['privacy', 'security', 'patient', 'phi', 'breach', 'authorization'],
        'gdpr': ['data protection', 'consent', 'privacy', 'data subject', 'processing']
    }
    
    # Add framework-specific keywords
    if framework in framework_keywords:
        for keyword in framework_keywords[framework]:
            if keyword in text_lower:
                keywords.append(keyword)
    
    # Add general compliance keywords
    general_keywords = ['compliance', 'regulation', 'requirement', 'documentation', 'evidence', 'control']
    for keyword in general_keywords:
        if keyword in text_lower:
            keywords.append(keyword)
    
    return list(set(keywords))  # Remove duplicates

def extract_sections(text_content: str) -> Dict[str, str]:
    """Extract sections from text content."""
    sections = {}
    
    # Simple section extraction based on common patterns
    lines = text_content.split('\n')
    current_section = 'main'
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line looks like a section header
        if (line.isupper() or 
            line.startswith('Section') or 
            line.startswith('Part') or
            line.startswith('Chapter') or
            len(line) < 100 and line.endswith(':')):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            # Start new section
            current_section = line.lower().replace(' ', '_').replace(':', '')[:50]
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def process_text_file(file_path: Path) -> ExtractedContent:
    """Process a single text file and extract structured content."""
    logger = logging.getLogger(__name__)
    
    try:
        # Read text content
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Detect framework
        framework = detect_regulatory_framework(text_content)
        
        # Classify document type
        doc_type = classify_document_type(text_content)
        
        # Extract keywords
        keywords = extract_keywords(text_content, framework)
        
        # Extract sections
        sections = extract_sections(text_content)
        
        # Create metadata
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            page_count=1,  # Text files are single "page"
            regulatory_framework=framework,
            document_type=doc_type,
            extraction_method="text_processor",
            confidence_score=0.9 if framework != 'unknown' else 0.5,
            processing_time=0.1
        )
        
        # Create extracted content
        extracted_content = ExtractedContent(
            text_content=text_content,
            tables=[],  # No tables in text files
            images=[],  # No images in text files
            metadata=metadata,
            sections=sections,
            keywords=keywords
        )
        
        logger.info(f"Processed {file_path.name} - Framework: {framework}, Type: {doc_type}")
        return extracted_content
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise

def process_directory(input_dir: str, output_dir: str, output_format: str = "json"):
    """Process all text files in a directory."""
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all text files
    text_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.pdf"))  # Our "PDFs" are actually text
    
    logger.info(f"Found {len(text_files)} text files to process")
    
    processed_count = 0
    failed_count = 0
    
    for file_path in text_files:
        try:
            # Process the file
            extracted_content = process_text_file(file_path)
            
            # Save processed content
            output_file = output_path / f"{file_path.stem}_processed.{output_format}"
            
            if output_format == "json":
                # Convert to JSON
                content_dict = {
                    "text_content": extracted_content.text_content,
                    "tables": extracted_content.tables,
                    "images": extracted_content.images,
                    "metadata": {
                        "filename": extracted_content.metadata.filename,
                        "file_size": extracted_content.metadata.file_size,
                        "page_count": extracted_content.metadata.page_count,
                        "regulatory_framework": extracted_content.metadata.regulatory_framework,
                        "document_type": extracted_content.metadata.document_type,
                        "extraction_method": extracted_content.metadata.extraction_method,
                        "confidence_score": extracted_content.metadata.confidence_score,
                        "processing_time": extracted_content.metadata.processing_time
                    },
                    "sections": extracted_content.sections,
                    "keywords": extracted_content.keywords
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(content_dict, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            logger.info(f"✅ Processed {file_path.name} -> {output_file.name}")
            
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ Failed to process {file_path.name}: {e}")
    
    logger.info(f"Processing complete. {processed_count} files processed, {failed_count} failed")
    return processed_count, failed_count

def main():
    """Main function."""
    logger = setup_logging()
    
    print("📄 Text Document Processor for Regulatory Compliance SLM System")
    print("="*60)
    
    input_dir = "data/raw"
    output_dir = "data/processed"
    output_format = "json"
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {output_format}")
    
    try:
        processed_count, failed_count = process_directory(input_dir, output_dir, output_format)
        
        print("\n" + "="*60)
        print("🎉 Processing Complete!")
        print("="*60)
        print(f"✅ Successfully processed: {processed_count}")
        print(f"❌ Failed: {failed_count}")
        
        if processed_count > 0:
            print(f"\n📁 Processed files saved to: {output_dir}")
            print("\n📚 Next steps:")
            print("1. Review processed files in data/processed/")
            print("2. Run: python3 model_trainer/src/train.py --data-path data/processed --output-path models/")
            print("3. Start training your SLM models!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 