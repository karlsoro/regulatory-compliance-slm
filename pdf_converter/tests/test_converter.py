"""
Unit tests for the PDF Converter application.

This module contains comprehensive tests for the PDF converter functionality,
including document processing, framework detection, and data extraction.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.converter import PDFConverter, DocumentMetadata, ExtractedContent


class TestPDFConverter:
    """Test cases for the PDFConverter class."""
    
    @pytest.fixture
    def converter(self):
        """Create a PDF converter instance for testing."""
        return PDFConverter(output_format="json")
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing."""
        return """
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
        """
    
    def test_converter_initialization(self, converter):
        """Test PDF converter initialization."""
        assert converter.output_format == "json"
        assert "pdfplumber" in converter.extraction_methods
        assert "pymupdf" in converter.extraction_methods
        assert "pypdf2" in converter.extraction_methods
        assert "ocr" in converter.extraction_methods
    
    def test_converter_invalid_format(self):
        """Test converter initialization with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            PDFConverter(output_format="invalid")
    
    def test_detect_regulatory_framework_sox(self, converter, sample_pdf_content):
        """Test SOX framework detection."""
        with patch('pdfplumber.open') as mock_pdf:
            mock_pdf.return_value.__enter__.return_value.pages = [
                Mock(extract_text=lambda: sample_pdf_content)
            ] * 3
            
            pdf_path = Path("test_sox.pdf")
            framework = converter._detect_regulatory_framework(pdf_path)
            assert framework == "sox"
    
    def test_detect_regulatory_framework_gaap(self, converter):
        """Test GAAP framework detection."""
        gaap_content = """
        Generally Accepted Accounting Principles
        
        Revenue Recognition - ASC 606
        
        The core principle of the revenue recognition standard is that an entity should 
        recognize revenue to depict the transfer of promised goods or services to customers 
        in an amount that reflects the consideration to which the entity expects to be 
        entitled in exchange for those goods or services.
        """
        
        with patch('pdfplumber.open') as mock_pdf:
            mock_pdf.return_value.__enter__.return_value.pages = [
                Mock(extract_text=lambda: gaap_content)
            ] * 3
            
            pdf_path = Path("test_gaap.pdf")
            framework = converter._detect_regulatory_framework(pdf_path)
            assert framework == "gaap"
    
    def test_detect_regulatory_framework_fda(self, converter):
        """Test FDA framework detection."""
        fda_content = """
        FDA 21 CFR Part 11
        
        Electronic Records; Electronic Signatures
        
        This part sets forth the criteria under which the agency considers electronic records, 
        electronic signatures, and handwritten signatures executed to electronic records to be 
        trustworthy, reliable, and generally equivalent to paper records and handwritten signatures.
        """
        
        with patch('pdfplumber.open') as mock_pdf:
            mock_pdf.return_value.__enter__.return_value.pages = [
                Mock(extract_text=lambda: fda_content)
            ] * 3
            
            pdf_path = Path("test_fda.pdf")
            framework = converter._detect_regulatory_framework(pdf_path)
            assert framework == "fda"
    
    def test_classify_document_type(self, converter):
        """Test document type classification."""
        # Test regulation document
        regulation_text = "This regulation establishes requirements for..."
        doc_type = converter._classify_document_type(regulation_text)
        assert doc_type == "regulation"
        
        # Test audit report
        audit_text = "Audit findings indicate several areas of concern..."
        doc_type = converter._classify_document_type(audit_text)
        assert doc_type == "audit_report"
        
        # Test policy document
        policy_text = "This policy outlines procedures for..."
        doc_type = converter._classify_document_type(policy_text)
        assert doc_type == "policy"
        
        # Test compliance document
        compliance_text = "Compliance checklist for internal controls..."
        doc_type = converter._classify_document_type(compliance_text)
        assert doc_type == "compliance_document"
    
    def test_extract_keywords_sox(self, converter):
        """Test keyword extraction for SOX framework."""
        text_content = "Internal controls for financial reporting must include..."
        keywords = converter._extract_keywords(text_content, "sox")
        
        expected_keywords = ["internal control", "financial reporting", "audit", "section 404"]
        for keyword in expected_keywords:
            assert keyword in keywords
    
    def test_extract_keywords_gaap(self, converter):
        """Test keyword extraction for GAAP framework."""
        text_content = "Financial statements must be prepared according to..."
        keywords = converter._extract_keywords(text_content, "gaap")
        
        expected_keywords = ["accounting", "financial statement", "revenue recognition"]
        for keyword in expected_keywords:
            assert keyword in keywords
    
    def test_extract_sections(self, converter):
        """Test section extraction from text."""
        text_content = """
        Section 1: Introduction
        This is the introduction section.
        
        Section 2: Requirements
        These are the requirements.
        
        Section 3: Conclusion
        This concludes the document.
        """
        
        sections = converter._extract_sections(text_content)
        
        assert "Section 1: Introduction" in sections
        assert "Section 2: Requirements" in sections
        assert "Section 3: Conclusion" in sections
        assert "This is the introduction section." in sections["Section 1: Introduction"]
    
    def test_get_page_count(self, converter):
        """Test page count extraction."""
        with patch('pdfplumber.open') as mock_pdf:
            mock_pdf.return_value.__enter__.return_value.pages = [Mock()] * 5
            
            pdf_path = Path("test.pdf")
            page_count = converter._get_page_count(pdf_path)
            assert page_count == 5
    
    @patch('pdfplumber.open')
    def test_extract_with_pdfplumber(self, mock_pdf, converter):
        """Test PDF extraction using pdfplumber."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample text content"
        mock_page.extract_tables.return_value = [["Header1", "Header2"], ["Data1", "Data2"]]
        mock_page.images = [{"name": "image1.png"}]
        mock_page.page_number = 1
        
        mock_pdf.return_value.__enter__.return_value.pages = [mock_page]
        
        pdf_path = Path("test.pdf")
        text, tables, images, confidence = converter._extract_with_pdfplumber(pdf_path)
        
        assert "Sample text content" in text
        assert len(tables) == 1
        assert tables[0]["data"] == [["Header1", "Header2"], ["Data1", "Data2"]]
        assert "image1.png" in images
        assert confidence > 0
    
    @patch('fitz.open')
    def test_extract_with_pymupdf(self, mock_fitz, converter):
        """Test PDF extraction using PyMuPDF."""
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample text content"
        mock_page.get_tables.return_value = [["Header1", "Header2"], ["Data1", "Data2"]]
        mock_page.get_images.return_value = [("image1", 0)]
        
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz.return_value = mock_doc
        
        pdf_path = Path("test.pdf")
        text, tables, images, confidence = converter._extract_with_pymupdf(pdf_path)
        
        assert "Sample text content" in text
        assert len(tables) == 1
        assert "image_0_image1" in images
        assert confidence > 0
    
    @patch('builtins.open')
    @patch('PyPDF2.PdfReader')
    def test_extract_with_pypdf2(self, mock_reader, mock_open, converter):
        """Test PDF extraction using PyPDF2."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample text content"
        mock_reader.return_value.pages = [mock_page]
        
        pdf_path = Path("test.pdf")
        text, tables, images, confidence = converter._extract_with_pypdf2(pdf_path)
        
        assert "Sample text content" in text
        assert len(tables) == 0  # PyPDF2 doesn't extract tables
        assert len(images) == 0  # PyPDF2 doesn't extract images
        assert confidence > 0
    
    def test_save_processed_data_json(self, converter):
        """Test saving processed data in JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Create sample content
            metadata = DocumentMetadata(
                filename="test.pdf",
                file_size=1024,
                page_count=5,
                regulatory_framework="sox",
                document_type="regulation",
                extraction_method="test",
                confidence_score=0.8,
                processing_time=1.5
            )
            
            content = ExtractedContent(
                text_content="Sample text content",
                tables=[],
                images=[],
                metadata=metadata,
                sections={"main": "Sample content"},
                keywords=["test", "sample"]
            )
            
            output_file = converter._save_processed_data(content, output_path, "test")
            
            assert output_file.exists()
            assert output_file.suffix == ".json"
            
            # Verify content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["text_content"] == "Sample text content"
            assert saved_data["metadata"]["regulatory_framework"] == "sox"
    
    def test_save_processed_data_csv(self):
        """Test saving processed data in CSV format."""
        converter = PDFConverter(output_format="csv")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            metadata = DocumentMetadata(
                filename="test.pdf",
                file_size=1024,
                page_count=5,
                regulatory_framework="sox",
                document_type="regulation",
                extraction_method="test",
                confidence_score=0.8,
                processing_time=1.5
            )
            
            content = ExtractedContent(
                text_content="Sample text content",
                tables=[],
                images=[],
                metadata=metadata,
                sections={"main": "Sample content"},
                keywords=["test", "sample"]
            )
            
            output_file = converter._save_processed_data(content, output_path, "test")
            
            assert output_file.exists()
            assert output_file.suffix == ".csv"
            
            # Verify content
            df = pd.read_csv(output_file)
            assert len(df) == 1
            assert df.iloc[0]["regulatory_framework"] == "sox"
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_process_folder_no_files(self, mock_glob, mock_exists, converter):
        """Test processing folder with no PDF files."""
        mock_exists.return_value = True
        mock_glob.return_value = []  # No PDF files
        
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = converter.process_folder(temp_dir, temp_dir)
            
            assert summary["processed"] == 0
            assert summary["failed"] == 0
            assert summary["total_files"] == 0
    
    @patch('pathlib.Path.exists')
    def test_process_folder_input_not_exists(self, mock_exists, converter):
        """Test processing folder that doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            converter.process_folder("nonexistent", "output")
    
    def test_process_single_file_not_exists(self, converter):
        """Test processing single file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            converter.process_single_file("nonexistent.pdf")


class TestDocumentMetadata:
    """Test cases for DocumentMetadata dataclass."""
    
    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_size=1024,
            page_count=5,
            regulatory_framework="sox",
            document_type="regulation",
            extraction_method="test",
            confidence_score=0.8,
            processing_time=1.5
        )
        
        assert metadata.filename == "test.pdf"
        assert metadata.file_size == 1024
        assert metadata.page_count == 5
        assert metadata.regulatory_framework == "sox"
        assert metadata.confidence_score == 0.8


class TestExtractedContent:
    """Test cases for ExtractedContent dataclass."""
    
    def test_extracted_content_creation(self):
        """Test ExtractedContent creation."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_size=1024,
            page_count=5,
            regulatory_framework="sox",
            document_type="regulation",
            extraction_method="test",
            confidence_score=0.8,
            processing_time=1.5
        )
        
        content = ExtractedContent(
            text_content="Sample text",
            tables=[],
            images=[],
            metadata=metadata,
            sections={"main": "Sample"},
            keywords=["test"]
        )
        
        assert content.text_content == "Sample text"
        assert content.metadata.regulatory_framework == "sox"
        assert len(content.keywords) == 1
        assert content.keywords[0] == "test" 