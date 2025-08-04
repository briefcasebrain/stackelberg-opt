#!/bin/bash
# Script to set up environment and run all tests for stackelberg-opt

set -e  # Exit on error

echo "=========================================="
echo "Setting up stackelberg-opt test environment"
echo "=========================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to project root
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Clean up any existing virtual environment
if [ -d "test_env" ]; then
    echo "Removing existing test environment..."
    rm -rf test_env
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv test_env

# Activate virtual environment
echo "Activating virtual environment..."
source test_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing stackelberg-opt in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Download spaCy model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Run all tests using the unified test runner
echo ""
echo "=========================================="
echo "Running all tests"
echo "=========================================="
python tests/run_all_tests.py --mode all

# Generate coverage report
echo ""
echo "=========================================="
echo "Generating coverage report"
echo "=========================================="
coverage html

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "✅ All tests completed!"
echo "📊 Coverage report generated in htmlcov/"
echo "🔍 View detailed report: open htmlcov/index.html"
echo ""
echo "To deactivate the virtual environment, run: deactivate"