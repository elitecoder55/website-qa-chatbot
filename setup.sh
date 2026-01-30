#!/bin/bash

# Setup script for Website QA Chatbot
# This script automates the setup process

echo "🤖 Website QA Chatbot - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check if Python 3.8+
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.8 or higher is required"
    exit 1
fi

echo "✅ Python version is compatible"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists"
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created (please add your OpenAI API key if you have one)"
else
    echo "ℹ️  .env file already exists"
fi
echo ""

# Create vector_store directory
echo "📁 Creating vector store directory..."
mkdir -p vector_store
echo "✅ Vector store directory created"
echo ""

echo "🎉 Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: streamlit run app.py"
echo ""
echo "Optional: Add your OpenAI API key to .env file for enhanced answers"
echo ""
