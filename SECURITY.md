# Security Guidelines

## 🔒 Critical: Protecting API Keys and Credentials

This project uses sensitive API keys for Galileo and OpenAI. **Never commit these to version control.**

### ✅ What's Protected

The `.gitignore` file is configured to prevent committing:

#### API Keys & Secrets
- `.env` and all `.env.*` files (except `.env.example`)
- Any file containing `api_key`, `apikey`, `secret`, `credentials`, `token`
- Cloud provider credentials (AWS, GCP, Azure)
- SSH keys (id_rsa, id_ed25519, etc.)
- PGP/GPG keys
- Certificate files (.pem, .key, .crt, etc.)

#### Database Files
- `.sql`, `.sqlite`, `.db` files
- Database configuration with credentials

#### Results & Logs
- `evaluation_results_*.json` (may contain API responses)
- All `.log` files (may contain sensitive debug info)
- Output directories

#### Configuration Files
- Local config overrides (`*.local.*`)
- Terraform state files
- Docker override files

### 🚨 Before Committing

**ALWAYS run these checks:**

1. **Never edit .env directly in the repo**
   ```bash
   # ✅ CORRECT
   cp .env.example .env
   # Edit .env locally (it's gitignored)

   # ❌ WRONG
   # Don't edit .env.example with real keys
   ```

2. **Check what you're committing**
   ```bash
   git status
   git diff
   ```

3. **Test the gitignore**
   ```bash
   bash test_gitignore.sh
   ```

4. **Verify no secrets in files**
   ```bash
   # Search for potential API keys in staged files
   git diff --cached | grep -i "api.*key\|secret\|password"
   ```

### 🔍 Pre-Commit Checklist

Before running `git commit`:

- [ ] No `.env` file in `git status`
- [ ] No API keys in code files
- [ ] No hardcoded credentials
- [ ] No evaluation results with sensitive data
- [ ] No log files with debug output
- [ ] `.env.example` has placeholder values only

### ⚠️ If You Accidentally Commit Secrets

**DO NOT just delete the file and commit again** - it's still in git history!

1. **Immediately rotate the exposed keys**
   - Generate new Galileo API key
   - Generate new OpenAI API key

2. **Remove from git history**
   ```bash
   # Option 1: Use BFG Repo Cleaner
   bfg --delete-files .env

   # Option 2: Use git filter-branch
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all

   # Force push (WARNING: This rewrites history)
   git push origin --force --all
   ```

3. **Update documentation**
   - Document the incident
   - Update keys immediately
   - Review access logs

### 🛡️ Best Practices

#### 1. Use Environment Variables
```python
# ✅ CORRECT
import os
api_key = os.getenv("OPENAI_API_KEY")

# ❌ WRONG
api_key = "sk-proj-abc123..."  # NEVER hardcode
```

#### 2. Keep .env.example Updated
```ini
# .env.example - Safe to commit
GALILEO_API_KEY=your-galileo-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here

# .env - NEVER commit
GALILEO_API_KEY=gal_live_abc123...
OPENAI_API_KEY=sk-proj-xyz789...
TAVILY_API_KEY=tavily_live_123abc...
```

#### 3. Review Code Before Committing
```bash
# Review all changes
git diff

# Review specific file
git diff tools.py

# Check what's staged
git diff --cached
```

#### 4. Use .gitignore Patterns
The `.gitignore` uses comprehensive patterns:
- `*api_key*` - Catches any filename with "api_key"
- `*.key` - Blocks all .key files
- `secrets.*` - Blocks any file starting with "secrets"

### 📋 Sensitive Data Checklist

Files that commonly contain secrets:

- [ ] `.env` files
- [ ] `config.py` or `settings.py`
- [ ] `credentials.json`
- [ ] Database connection strings
- [ ] Docker Compose files with passwords
- [ ] CI/CD configuration files
- [ ] Jupyter notebooks with API calls
- [ ] Test files with real API keys

### 🔧 Testing Security

Run the security test:
```bash
bash test_gitignore.sh
```

Expected output:
```
✅ SUCCESS: All sensitive files are ignored!
```

### 🚀 Safe Sharing

When sharing this project:

1. **Clone includes no secrets**
   ```bash
   git clone <repo>
   # .env is NOT included (gitignored)
   ```

2. **Recipients need to create .env**
   ```bash
   cp .env.example .env
   # Add their own API keys
   ```

3. **README includes setup instructions**
   - How to get API keys
   - How to configure .env
   - How to verify setup

### 📞 Questions?

- **What if I need to share config?** Use `.env.example` with placeholder values
- **How do I test locally?** Create `.env` from `.env.example`
- **Can I commit results?** Only if they contain no sensitive data (use `--quick` mode for testing)
- **What about logs?** All `.log` files are gitignored - use stdout for important info

### 🔗 Additional Resources

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [OWASP: Sensitive Data Exposure](https://owasp.org/www-community/vulnerabilities/Sensitive_Data_Exposure)
- [Git Secrets](https://github.com/awslabs/git-secrets) - Prevents committing secrets

---

**Remember**: It's easier to prevent a leak than to clean one up. Always double-check before committing!
