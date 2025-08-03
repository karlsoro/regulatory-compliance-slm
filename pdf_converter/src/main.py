#!/usr/bin/env python3
"""
Main entry point for the PDF Converter application.

This script provides a command-line interface for converting PDF documents
into structured training data for regulatory compliance SLMs.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from converter import PDFConverter


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pdf_converter.log')
        ]
    )


def main():
    """Main function for the PDF converter CLI."""
    parser = argparse.ArgumentParser(
        description="Convert PDF documents to training data for regulatory compliance SLMs"
    )
    
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing PDF files to process"
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save processed data"
    )
    
    parser.add_argument(
        "--regulatory-framework",
        choices=["sox", "gaap", "fda", "hipaa", "gdpr", "auto"],
        default="auto",
        help="Regulatory framework to assign to documents (default: auto-detect)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "parquet"],
        default="json",
        help="Output format for processed data (default: json)"
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
        help="Show what would be processed without actually processing files"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle regulatory framework
    regulatory_framework = None if args.regulatory_framework == "auto" else args.regulatory_framework
    
    logger.info(f"Starting PDF conversion process")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Output format: {args.output_format}")
    logger.info(f"Regulatory framework: {regulatory_framework or 'auto-detect'}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be processed")
        # Count PDF files
        pdf_files = list(input_path.glob("**/*.pdf"))
        logger.info(f"Would process {len(pdf_files)} PDF files")
        return
    
    try:
        # Initialize converter
        converter = PDFConverter(output_format=args.output_format)
        
        # Process files
        summary = converter.process_folder(
            input_dir=str(input_path),
            output_dir=str(output_path),
            regulatory_framework=regulatory_framework
        )
        
        # Print summary
        logger.info("Processing complete!")
        logger.info(f"Total files: {summary['total_files']}")
        logger.info(f"Successfully processed: {summary['processed']}")
        logger.info(f"Failed: {summary['failed']}")
        
        if summary['failed'] > 0:
            logger.warning("Some files failed to process. Check the log for details.")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 