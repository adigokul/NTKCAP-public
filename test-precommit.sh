#!/bin/bash
# Test script to validate pre-commit hook functionality

echo "Testing pre-commit hook..."
echo ""

# Run the pre-commit hook directly
if [ -f "git-hooks/pre-commit" ]; then
    echo "Executing pre-commit hook:"
    bash git-hooks/pre-commit
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ Pre-commit hook executed successfully!"
        echo "Current Git configuration:"
        echo "  Name:  $(git config user.name)"
        echo "  Email: $(git config user.email)"
    else
        echo ""
        echo "❌ Pre-commit hook failed with exit code: $exit_code"
    fi
else
    echo "❌ Pre-commit hook not found at git-hooks/pre-commit"
fi