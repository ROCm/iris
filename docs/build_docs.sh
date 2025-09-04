#!/bin/bash

# Iris Documentation Build Script
# This script builds the Iris documentation using Sphinx

set -e  # Exit on any error

echo "🚀 Building Iris Documentation..."

# Check if we're in the right directory
if [ ! -f "conf.py" ]; then
    echo "❌ Error: Please run this script from the docs/ directory"
    echo "   cd docs && ./build_docs.sh"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📚 Installing documentation dependencies..."
pip install -r sphinx/requirements.txt

# Create build directories
echo "📁 Creating build directories..."
rm -rf _build
mkdir -p _build/html _build/doctrees

# Build the documentation
echo "🔨 Building documentation..."
python3 -m sphinx -b html -d _build/doctrees -D language=en . _build/html

# Copy images to build directory
echo "🖼️ Copying images to build directory..."
mkdir -p _build/html/images
cp ../images/*.png _build/html/images/ 2>/dev/null || echo "Warning: Could not copy images automatically"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Documentation built successfully!"
    echo ""
    echo "📖 You can now view the documentation by:"
    echo "   1. Opening _build/html/index.html in your browser"
    echo "   2. Running: python3 -m http.server -d _build/html/"
    echo "   3. Then visiting: http://localhost:8000"
    echo ""
    echo "🚀 To serve the docs automatically:"
    echo "   python3 -m sphinx_autobuild -b html -d _build/doctrees -D language=en . _build/html"
else
    echo "❌ Documentation build failed!"
    exit 1
fi
