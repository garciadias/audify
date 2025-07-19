#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
uv sync

# Create data directory if it doesn't exist
mkdir -p data/output

echo "Setup complete!"
echo ""
echo "To run the application:"
echo "  uv run python main.py"
echo ""
echo "Then open your browser to: http://localhost:8000"
