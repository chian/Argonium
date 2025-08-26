#!/bin/bash

# Argonium Environment Setup Script
echo "Setting up Argonium environment variables..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# OpenAI API Key (required for gpt41 and gpt35)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional, for claude3)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Semantic Scholar API Key (optional, for paper discovery)
SS_API_KEY=your_semantic_scholar_api_key_here
EOF
    echo "Created .env file. Please edit it with your actual API keys."
else
    echo ".env file already exists."
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment variables loaded from .env"
fi

echo ""
echo "To set up your environment:"
echo "1. Edit the .env file with your actual API keys"
echo "2. Or set environment variables directly:"
echo "   export OPENAI_API_KEY='your_key_here'"
echo "3. Test with: python make_v21.py --help"
