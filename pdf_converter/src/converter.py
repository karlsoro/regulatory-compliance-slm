"""
PDF Converter for Regulatory Compliance Training Data

This module provides functionality to convert PDF documents into structured
training data for small language models focused on regulatory compliance.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import pdfplumber
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    filename: str
    file_size: int
    page_count: int
    regulatory_framework: str
    document_type: str
    extraction_method: str
    confidence_score: float
    processing_time: float


@dataclass
class ExtractedContent:
    """Structured content extracted from PDF documents."""
    text_content: str
    tables: List[Dict[str, Any]]
    images: List[str]
    metadata: DocumentMetadata
    sections: Dict[str, str]
    keywords: List[str]


class PDFConverter:
    """
    Converts PDF documents into structured training data for regulatory compliance SLMs.
    
    This converter handles various types of regulatory documents including:
    - Official regulations (SOX, GAAP, FDA guidelines)
    - Company compliance documents
    - Audit reports and findings
    - Internal control documentation
    """
    
    def __init__(self, output_format: str = "json"):
        """
        Initialize the PDF converter.
        
        Args:
            output_format: Output format for processed data ("json", "csv", "parquet")
        """
        self.output_format = output_format
        self.supported_formats = ["json", "csv", "parquet"]
        
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}")
        
        # Initialize extraction methods
        self.extraction_methods = {
            "pdfplumber": self._extract_with_pdfplumber,
            "pymupdf": self._extract_with_pymupdf,
            "pypdf2": self._extract_with_pypdf2,
            "ocr": self._extract_with_ocr
        }
        
        logger.info(f"PDF Converter initialized with output format: {output_format}")
    
    def process_folder(self, input_dir: str, output_dir: str, 
                      regulatory_framework: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all PDF files in a folder and convert them to training data.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed data
            regulatory_framework: Specific regulatory framework (e.g., "sox", "fda")
            
        Returns:
            Dictionary containing processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return {"processed": 0, "failed": 0, "files": []}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed_files = []
        failed_files = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                # Determine regulatory framework if not specified
                framework = regulatory_framework or self._detect_regulatory_framework(pdf_file)
                
                # Process the file
                extracted_content = self.process_single_file(
                    str(pdf_file), 
                    regulatory_framework=framework
                )
                
                # Save processed data
                output_file = self._save_processed_data(
                    extracted_content, 
                    output_path, 
                    pdf_file.stem
                )
                
                processed_files.append({
                    "input_file": str(pdf_file),
                    "output_file": str(output_file),
                    "framework": framework,
                    "metadata": asdict(extracted_content.metadata)
                })
                
                logger.debug(f"Successfully processed: {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                failed_files.append({"file": str(pdf_file), "error": str(e)})
        
        # Save processing summary
        summary = {
            "total_files": len(pdf_files),
            "processed": len(processed_files),
            "failed": len(failed_files),
            "processed_files": processed_files,
            "failed_files": failed_files
        }
        
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete. {len(processed_files)} files processed, "
                   f"{len(failed_files)} failed")
        
        return summary
    
    def process_single_file(self, pdf_path: str, 
                           regulatory_framework: Optional[str] = None) -> ExtractedContent:
        """
        Process a single PDF file and extract structured content.
        
        Args:
            pdf_path: Path to the PDF file
            regulatory_framework: Regulatory framework identifier
            
        Returns:
            ExtractedContent object with structured data
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get file metadata
        file_size = pdf_path.stat().st_size
        page_count = self._get_page_count(pdf_path)
        
        # Detect regulatory framework if not provided
        if not regulatory_framework:
            regulatory_framework = self._detect_regulatory_framework(pdf_path)
        
        # Extract content using multiple methods
        text_content, tables, images, confidence_score = self._extract_content(pdf_path)
        
        # Extract sections and keywords
        sections = self._extract_sections(text_content)
        keywords = self._extract_keywords(text_content, regulatory_framework)
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata = DocumentMetadata(
            filename=pdf_path.name,
            file_size=file_size,
            page_count=page_count,
            regulatory_framework=regulatory_framework,
            document_type=self._classify_document_type(text_content),
            extraction_method="multi_method",
            confidence_score=confidence_score,
            processing_time=processing_time
        )
        
        return ExtractedContent(
            text_content=text_content,
            tables=tables,
            images=images,
            metadata=metadata,
            sections=sections,
            keywords=keywords
        )
    
    def _extract_content(self, pdf_path: Path) -> Tuple[str, List[Dict], List[str], float]:
        """Extract content using multiple methods for best results."""
        all_text = []
        all_tables = []
        all_images = []
        confidence_scores = []
        
        # Try different extraction methods
        for method_name, method_func in self.extraction_methods.items():
            try:
                text, tables, images, confidence = method_func(pdf_path)
                all_text.append(text)
                all_tables.extend(tables)
                all_images.extend(images)
                confidence_scores.append(confidence)
                logger.debug(f"Method {method_name} extracted {len(text)} characters")
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {str(e)}")
                confidence_scores.append(0.0)
        
        # Combine results (use best text extraction)
        if all_text:
            best_text = max(all_text, key=len)
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            best_text = ""
            avg_confidence = 0.0
        
        return best_text, all_tables, all_images, avg_confidence
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, List[Dict], List[str], float]:
        """Extract content using pdfplumber."""
        text_content = ""
        tables = []
        images = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"
                
                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        tables.append({
                            "data": table,
                            "page": page.page_number
                        })
                
                # Extract images (basic detection)
                if page.images:
                    images.extend([img["name"] for img in page.images])
        
        confidence = min(1.0, len(text_content) / 1000)  # Simple confidence metric
        return text_content, tables, images, confidence
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, List[Dict], List[str], float]:
        """Extract content using PyMuPDF."""
        text_content = ""
        tables = []
        images = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            page_text = page.get_text()
            text_content += page_text + "\n"
            
            # Extract tables (basic detection)
            tables_on_page = page.get_tables()
            for table in tables_on_page:
                tables.append({
                    "data": table,
                    "page": page_num + 1
                })
            
            # Extract images
            image_list = page.get_images()
            for img in image_list:
                images.append(f"image_{page_num}_{img[0]}")
        
        doc.close()
        
        confidence = min(1.0, len(text_content) / 1000)
        return text_content, tables, images, confidence
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Tuple[str, List[Dict], List[str], float]:
        """Extract content using PyPDF2."""
        text_content = ""
        tables = []
        images = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"
        
        confidence = min(1.0, len(text_content) / 1000)
        return text_content, tables, images, confidence
    
    def _extract_with_ocr(self, pdf_path: Path) -> Tuple[str, List[Dict], List[str], float]:
        """Extract content using OCR (for scanned documents)."""
        # This is a simplified OCR implementation
        # In production, you might want to use more sophisticated OCR
        text_content = ""
        tables = []
        images = []
        
        # Convert PDF to images and perform OCR
        try:
            from pdf2image import convert_from_path
            images_pil = convert_from_path(pdf_path)
            
            for i, image in enumerate(images_pil):
                # Perform OCR
                page_text = pytesseract.image_to_string(image)
                text_content += page_text + "\n"
                images.append(f"page_{i+1}")
        except Exception as e:
            logger.warning(f"OCR extraction failed: {str(e)}")
        
        confidence = 0.7  # OCR typically has lower confidence
        return text_content, tables, images, confidence
    
    def _detect_regulatory_framework(self, pdf_path: Path) -> str:
        """Detect regulatory framework based on content and filename."""
        # Read first few pages to detect framework
        try:
            with pdfplumber.open(pdf_path) as pdf:
                sample_text = ""
                for page in pdf.pages[:3]:  # Check first 3 pages
                    text = page.extract_text() or ""
                    sample_text += text + " "
        except:
            sample_text = pdf_path.name.lower()
        
        # Framework detection logic
        sample_text_lower = sample_text.lower()
        
        if any(keyword in sample_text_lower for keyword in ["sox", "sarbanes", "oxley"]):
            return "sox"
        elif any(keyword in sample_text_lower for keyword in ["gaap", "accounting", "financial"]):
            return "gaap"
        elif any(keyword in sample_text_lower for keyword in ["fda", "21 cfr", "drug", "medical"]):
            return "fda"
        elif any(keyword in sample_text_lower for keyword in ["hipaa", "healthcare", "privacy"]):
            return "hipaa"
        elif any(keyword in sample_text_lower for keyword in ["gdpr", "data protection", "privacy"]):
            return "gdpr"
        else:
            return "unknown"
    
    def _classify_document_type(self, text_content: str) -> str:
        """Classify the type of document based on content."""
        text_lower = text_content.lower()
        
        if any(keyword in text_lower for keyword in ["regulation", "rule", "act"]):
            return "regulation"
        elif any(keyword in text_lower for keyword in ["audit", "report", "finding"]):
            return "audit_report"
        elif any(keyword in text_lower for keyword in ["policy", "procedure", "guideline"]):
            return "policy"
        elif any(keyword in text_lower for keyword in ["control", "compliance", "checklist"]):
            return "compliance_document"
        else:
            return "general"
    
    def _extract_sections(self, text_content: str) -> Dict[str, str]:
        """Extract document sections based on headers."""
        sections = {}
        lines = text_content.split('\n')
        current_section = "main"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers (simplified)
            if (line.isupper() or 
                line.startswith('Section ') or 
                line.startswith('Chapter ') or
                len(line) < 100 and line.endswith(':')):
                
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                current_section = line.replace(':', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_keywords(self, text_content: str, framework: str) -> List[str]:
        """Extract relevant keywords based on regulatory framework."""
        # Framework-specific keywords
        framework_keywords = {
            "sox": ["internal control", "financial reporting", "audit", "section 404", 
                   "section 302", "disclosure", "material weakness"],
            "gaap": ["accounting", "financial statement", "revenue recognition", 
                    "asset", "liability", "equity", "income statement"],
            "fda": ["clinical trial", "drug approval", "manufacturing", "quality control",
                   "21 cfr", "electronic record", "validation"],
            "hipaa": ["privacy", "security", "patient", "phi", "breach", "authorization",
                     "minimum necessary"],
            "gdpr": ["data protection", "consent", "right to be forgotten", "data subject",
                    "controller", "processor"]
        }
        
        keywords = framework_keywords.get(framework, [])
        
        # Add common compliance keywords
        common_keywords = ["compliance", "regulation", "requirement", "documentation", 
                          "evidence", "process", "procedure", "policy"]
        keywords.extend(common_keywords)
        
        return keywords
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except:
            return 0
    
    def _save_processed_data(self, content: ExtractedContent, 
                           output_path: Path, filename: str) -> Path:
        """Save processed data in the specified format."""
        if self.output_format == "json":
            output_file = output_path / f"{filename}.json"
            with open(output_file, 'w') as f:
                json.dump(asdict(content), f, indent=2, default=str)
        
        elif self.output_format == "csv":
            # Save text content and metadata as CSV
            output_file = output_path / f"{filename}.csv"
            data = {
                "text_content": [content.text_content],
                "regulatory_framework": [content.metadata.regulatory_framework],
                "document_type": [content.metadata.document_type],
                "page_count": [content.metadata.page_count],
                "confidence_score": [content.metadata.confidence_score]
            }
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
        
        elif self.output_format == "parquet":
            output_file = output_path / f"{filename}.parquet"
            data = {
                "text_content": [content.text_content],
                "regulatory_framework": [content.metadata.regulatory_framework],
                "document_type": [content.metadata.document_type],
                "page_count": [content.metadata.page_count],
                "confidence_score": [content.metadata.confidence_score]
            }
            df = pd.DataFrame(data)
            df.to_parquet(output_file, index=False)
        
        return output_file 