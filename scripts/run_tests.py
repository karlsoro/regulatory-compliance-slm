#!/usr/bin/env python3
"""
Test Runner for Regulatory Compliance SLM System

This script provides a comprehensive test runner that can execute all types of tests
and generate detailed reports for the regulatory compliance SLM system.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime


def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return {
            "command": command,
            "description": description,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "success": result.returncode == 0
        }
        
    except Exception as e:
        print(f"Error running command: {e}")
        return {
            "command": command,
            "description": description,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "duration": 0,
            "success": False
        }


def run_unit_tests():
    """Run unit tests for all components."""
    results = []
    
    # Test PDF Converter
    results.append(run_command(
        ["python", "-m", "pytest", "pdf_converter/tests/", "-v", "--tb=short"],
        "PDF Converter Unit Tests"
    ))
    
    # Test Model Trainer
    results.append(run_command(
        ["python", "-m", "pytest", "model_trainer/tests/", "-v", "--tb=short"],
        "Model Trainer Unit Tests"
    ))
    
    # Test Broker
    results.append(run_command(
        ["python", "-m", "pytest", "broker/tests/", "-v", "--tb=short"],
        "Broker Unit Tests"
    ))
    
    return results


def run_integration_tests():
    """Run integration tests."""
    results = []
    
    # Run integration tests
    results.append(run_command(
        ["python", "-m", "pytest", "tests/integration/", "-v", "-m", "integration"],
        "Integration Tests"
    ))
    
    return results


def run_e2e_tests():
    """Run end-to-end tests."""
    results = []
    
    # Run end-to-end tests
    results.append(run_command(
        ["python", "-m", "pytest", "tests/e2e/", "-v", "-m", "e2e"],
        "End-to-End Tests"
    ))
    
    return results


def run_all_tests():
    """Run all tests."""
    results = []
    
    # Run all tests with coverage
    results.append(run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=html", "--cov-report=term"],
        "All Tests with Coverage"
    ))
    
    return results


def run_specific_component_tests(component):
    """Run tests for a specific component."""
    component_tests = {
        "pdf_converter": "pdf_converter/tests/",
        "model_trainer": "model_trainer/tests/",
        "broker": "broker/tests/",
        "integration": "tests/integration/",
        "e2e": "tests/e2e/"
    }
    
    if component not in component_tests:
        print(f"Unknown component: {component}")
        print(f"Available components: {list(component_tests.keys())}")
        return []
    
    return [run_command(
        ["python", "-m", "pytest", component_tests[component], "-v"],
        f"{component.title()} Tests"
    )]


def generate_test_report(results, output_file="test_report.json"):
    """Generate a test report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "total_duration": sum(r["duration"] for r in results)
        },
        "results": results
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total test suites: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Total duration: {report['summary']['total_duration']:.2f} seconds")
    
    # Print failed tests
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print(f"\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test['description']}: {test['stderr']}")
    
    return report


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-cov",
        "torch",
        "transformers",
        "datasets",
        "fastapi",
        "uvicorn",
        "pdfplumber",
        "pymupdf",
        "pandas",
        "numpy",
        "scikit-learn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All required dependencies are installed.")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test Runner for Regulatory Compliance SLM System"
    )
    
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "e2e", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--component",
        choices=["pdf_converter", "model_trainer", "broker", "integration", "e2e"],
        help="Run tests for specific component"
    )
    
    parser.add_argument(
        "--report",
        default="test_report.json",
        help="Output file for test report (default: test_report.json)"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies before running tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
    
    # Run tests based on arguments
    if args.component:
        results = run_specific_component_tests(args.component)
    elif args.type == "unit":
        results = run_unit_tests()
    elif args.type == "integration":
        results = run_integration_tests()
    elif args.type == "e2e":
        results = run_e2e_tests()
    else:  # all
        results = run_all_tests()
    
    # Generate report
    report = generate_test_report(results, args.report)
    
    # Exit with appropriate code
    if report["summary"]["failed"] > 0:
        print(f"\n❌ {report['summary']['failed']} test suite(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All {report['summary']['total_tests']} test suite(s) passed")
        sys.exit(0)


if __name__ == "__main__":
    main() 