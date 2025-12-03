#!/bin/bash

# Script to set up final_project as a new git repository with multiple remotes
# Usage: ./setup_git_repo.sh

set -e

echo "Setting up final_project as a new git repository..."

# Navigate to final_project directory
cd "$(dirname "$0")"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Check current remotes
echo ""
echo "Current remotes:"
git remote -v

echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo ""
echo "1. Create a new repository on your personal GitHub account"
echo "   (e.g., https://github.com/YOUR_USERNAME/final-project)"
echo ""
echo "2. Add your personal GitHub as a remote:"
echo "   git remote add personal https://github.com/YOUR_USERNAME/final-project.git"
echo ""
echo "3. (Optional) Add the original repo as another remote:"
echo "   git remote add origin https://github.com/umass-CS690F/proj-group-04.git"
echo ""
echo "4. Stage and commit your files:"
echo "   git add ."
echo "   git commit -m 'Initial commit: Final project'"
echo ""
echo "5. Push to your personal GitHub:"
echo "   git push -u personal main"
echo ""
echo "6. (Optional) Push to both remotes:"
echo "   git push personal main"
echo "   git push origin main"
echo ""
echo "=========================================="
echo ""
echo "To push to both remotes at once, you can use:"
echo "  git push personal main && git push origin main"
echo ""
echo "Or set up a custom push command in .git/config:"
echo "  [remote \"all\"]"
echo "    url = https://github.com/YOUR_USERNAME/final-project.git"
echo "    url = https://github.com/umass-CS690F/proj-group-04.git"
echo ""

