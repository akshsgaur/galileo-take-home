#!/bin/bash
# Test script to verify .gitignore is working correctly

echo "=================================="
echo "Testing .gitignore Configuration"
echo "=================================="
echo ""

# Create test files that should be ignored
echo "Creating test files that should be ignored..."

# API key files
touch .env.test
touch galileo_key.txt
touch api_key.json
touch credentials.json

# Log files
touch test.log
touch debug.out

# Results
touch evaluation_results_12345.json

# Python cache
mkdir -p __pycache__
touch __pycache__/test.pyc

echo "✓ Test files created"
echo ""

# Check git status
echo "Checking git status (these files should NOT appear)..."
echo "================================================"

if command -v git &> /dev/null; then
    # Initialize git if not already
    if [ ! -d .git ]; then
        git init
        echo "✓ Git repository initialized"
    fi

    # Check what would be committed
    git status --short

    # Count untracked files
    UNTRACKED=$(git status --short | grep "^??" | wc -l | tr -d ' ')

    echo ""
    echo "================================================"
    if [ "$UNTRACKED" -eq "0" ]; then
        echo "✅ SUCCESS: All sensitive files are ignored!"
    else
        echo "⚠️  WARNING: $UNTRACKED sensitive files are not ignored!"
        echo ""
        echo "Untracked files that should be ignored:"
        git status --short | grep "^??"
    fi
else
    echo "Git not installed. Skipping git status check."
fi

echo ""
echo "Cleaning up test files..."
rm -f .env.test galileo_key.txt api_key.json credentials.json
rm -f test.log debug.out evaluation_results_12345.json
rm -rf __pycache__

echo "✓ Cleanup complete"
echo ""
echo "=================================="
echo "Test Complete"
echo "=================================="
