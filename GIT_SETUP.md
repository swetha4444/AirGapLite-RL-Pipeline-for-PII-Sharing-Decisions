# Git Setup Guide: Push final_project to Multiple GitHub Repositories

This guide will help you set up `final_project` as a separate git repository that can be pushed to both your personal GitHub and the original course repository.

## Step 1: Create a New Repository on Your Personal GitHub

1. Go to https://github.com/new
2. Create a new repository (e.g., `final-project` or `pii-minimization-project`)
3. **Do NOT** initialize it with a README, .gitignore, or license (we already have these)
4. Copy the repository URL (e.g., `https://github.com/YOUR_USERNAME/final-project.git`)

## Step 2: Add Multiple Remotes

Run these commands in the `final_project` directory:

```bash
cd final_project

# Add your personal GitHub as 'personal' remote
git remote add personal https://github.com/YOUR_USERNAME/final-project.git

# Add the original course repo as 'origin' remote (optional)
git remote add origin https://github.com/umass-CS690F/proj-group-04.git
```

**Note:** Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 3: Stage and Commit Files

```bash
# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Final project - PII Minimization"
```

## Step 4: Push to Your Personal GitHub

```bash
# Push to your personal GitHub (main branch)
git push -u personal main
```

If your personal repo uses a different default branch (e.g., `master`), use:
```bash
git push -u personal master
```

## Step 5: Push to Both Repositories (Optional)

### Option A: Push to both separately
```bash
git push personal main
git push origin main
```

### Option B: Set up a single remote that pushes to both

Edit `.git/config` and add:
```ini
[remote "all"]
    url = https://github.com/YOUR_USERNAME/final-project.git
    url = https://github.com/umass-CS690F/proj-group-04.git
```

Then push to both with:
```bash
git push all main
```

### Option C: Use a shell alias/function

Add to your `~/.zshrc` or `~/.bashrc`:
```bash
alias git-push-all='git push personal main && git push origin main'
```

Then use:
```bash
git-push-all
```

## Verify Your Setup

Check your remotes:
```bash
git remote -v
```

You should see:
```
origin    https://github.com/umass-CS690F/proj-group-04.git (fetch)
origin    https://github.com/umass-CS690F/proj-group-04.git (push)
personal  https://github.com/YOUR_USERNAME/final-project.git (fetch)
personal  https://github.com/YOUR_USERNAME/final-project.git (push)
```

## Future Updates

When you make changes and want to push to both repositories:

```bash
# Make your changes, then:
git add .
git commit -m "Your commit message"
git push personal main
git push origin main
```

Or if you set up the "all" remote:
```bash
git push all main
```

## Troubleshooting

### If you get authentication errors:
- Make sure you're authenticated with GitHub CLI: `gh auth login`
- Or use SSH URLs instead of HTTPS:
  - `git remote set-url personal git@github.com:YOUR_USERNAME/final-project.git`
  - `git remote set-url origin git@github.com:umass-CS690F/proj-group-04.git`

### If the branch name is different:
- Check your default branch: `git branch`
- Use the correct branch name in push commands

### If you want to remove a remote:
```bash
git remote remove personal  # or origin
```

