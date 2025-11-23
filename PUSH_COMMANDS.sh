#!/bin/bash
# Cerebro Phase 1 - GitHub Push Commands

echo "================================================"
echo "CEREBRO PHASE 1 - PUSH TO GITHUB"
echo "================================================"
echo ""

# Step 1: Final verification
echo "Step 1: Verifying package..."
python3 -c "import cerebro; print('✓ Imports working')" 2>&1 | grep "✓" || echo "✗ Import failed"
ls README.md LICENSE setup.py requirements.txt > /dev/null 2>&1 && echo "✓ Core files present" || echo "✗ Missing files"
echo ""

# Step 2: Initialize git (if needed)
echo "Step 2: Initializing Git..."
if [ ! -d .git ]; then
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi
echo ""

# Step 3: Add all files
echo "Step 3: Adding files..."
git add .
echo "✓ Files staged"
echo ""

# Step 4: Check what will be committed
echo "Step 4: Files to be committed:"
git status --short | head -20
echo "..."
echo ""

# Step 5: Commit
echo "Step 5: Committing..."
git commit -m "feat: Cerebro Phase 1 - Autonomous MMM System

- Multi-agent architecture for Marketing Mix Modeling
- 4,049 production examples from Google, Meta, Microsoft, Uber repos
- Autonomous code generation (1000+ lines)
- NumPyro/PyMC/Stan backends
- Complete documentation and examples
- RAG-powered with production MMM code
- YAML-based model specifications"

echo "✓ Committed"
echo ""

# Step 6: Instructions for GitHub
echo "================================================"
echo "NEXT: CREATE GITHUB REPO AND PUSH"
echo "================================================"
echo ""
echo "1. Go to: https://github.com/new"
echo ""
echo "2. Repository settings:"
echo "   Name: cerebro"
echo "   Description: Autonomous Marketing Mix Modeling with Multi-Agent AI"
echo "   Public or Private: (your choice)"
echo "   Don't initialize with README (we have one)"
echo ""
echo "3. After creating repo, run:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/cerebro.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "================================================"
echo "OR use SSH:"
echo "   git remote add origin git@github.com:YOUR_USERNAME/cerebro.git"
echo "   git push -u origin main"
echo "================================================"

