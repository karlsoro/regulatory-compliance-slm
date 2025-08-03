#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the project structure and basic functionality
without requiring external dependencies.
"""

import os
import sys
import json
from pathlib import Path


def test_project_structure():
    """Test that the project structure is correct."""
    print("Testing project structure...")
    
    required_dirs = [
        "pdf_converter/src",
        "pdf_converter/tests", 
        "model_trainer/src",
        "model_trainer/tests",
        "broker/src",
        "broker/tests",
        "shared/utils",
        "shared/config",
        "data/raw",
        "data/processed",
        "data/training",
        "models/checkpoints",
        "models/deployed",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "docs",
        "scripts"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "docker-compose.yml",
        "pdf_converter/src/__init__.py",
        "pdf_converter/src/converter.py",
        "pdf_converter/src/main.py",
        "model_trainer/src/__init__.py",
        "model_trainer/src/trainer.py",
        "model_trainer/src/train.py",
        "broker/src/__init__.py",
        "broker/src/broker.py",
        "broker/src/main.py",
        "shared/config/training_config.yaml",
        "scripts/run_tests.py",
        "scripts/demo.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
    
    return True


def test_basic_imports():
    """Test basic Python imports without external dependencies."""
    print("\nTesting basic imports...")
    
    try:
        # Test that we can import basic Python modules
        import tempfile
        import json
        import asyncio
        from pathlib import Path
        from dataclasses import dataclass
        from typing import Dict, List, Optional
        
        print("✅ Basic Python imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_config_files():
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")
    
    try:
        # Test YAML config
        config_path = Path("shared/config/training_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
                if "model_name" in content and "frameworks" in content:
                    print("✅ Training config file is valid")
                else:
                    print("❌ Training config file is invalid")
                    return False
        else:
            print("❌ Training config file not found")
            return False
        
        # Test requirements.txt
        req_path = Path("requirements.txt")
        if req_path.exists():
            with open(req_path, 'r') as f:
                content = f.read()
                if "torch" in content and "transformers" in content:
                    print("✅ Requirements file is valid")
                else:
                    print("❌ Requirements file is invalid")
                    return False
        else:
            print("❌ Requirements file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False


def test_git_setup():
    """Test that Git is properly set up."""
    print("\nTesting Git setup...")
    
    try:
        git_dir = Path(".git")
        if git_dir.exists():
            print("✅ Git repository initialized")
            
            # Check if we have commits
            import subprocess
            result = subprocess.run(
                ["git", "log", "--oneline"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                print("✅ Git has commits")
                return True
            else:
                print("❌ Git has no commits")
                return False
        else:
            print("❌ Git repository not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Git test error: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    print("\nCreating sample data...")
    
    try:
        # Create sample PDF data
        sample_data = {
            "sox_regulation": {
                "text_content": "Sarbanes-Oxley Act of 2002 - Section 404 requires internal controls for financial reporting.",
                "regulatory_framework": "sox",
                "document_type": "regulation",
                "page_count": 5,
                "confidence_score": 0.9
            },
            "gaap_guidelines": {
                "text_content": "Generally Accepted Accounting Principles - Revenue recognition standards for financial reporting.",
                "regulatory_framework": "gaap", 
                "document_type": "guideline",
                "page_count": 3,
                "confidence_score": 0.85
            }
        }
        
        # Save to data directory
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, data in sample_data.items():
            file_path = data_dir / f"{filename}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"✅ Created {len(sample_data)} sample data files")
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation error: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Regulatory Compliance SLM System Structure")
    print("="*60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Basic Imports", test_basic_imports),
        ("Configuration Files", test_config_files),
        ("Git Setup", test_git_setup),
        ("Sample Data Creation", create_sample_data)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project structure is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the demo: python3 scripts/demo.py")
        print("3. Run tests: python3 scripts/run_tests.py")
        print("4. Start development!")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 