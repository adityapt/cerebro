# Contributing to Cerebro

Thank you for your interest in contributing to Cerebro! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/cerebro.git
   cd cerebro
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install Ollama (for local testing)**
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull the model
   ollama pull qwen2.5:7b
   ```

## 🔧 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow PEP 8 style guidelines
- Add docstrings to new functions/classes
- Update tests if needed

### 3. Run Tests

```bash
pytest tests/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add: brief description of changes"
```

Use conventional commit messages:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for improvements
- `Docs:` for documentation

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 📝 Code Standards

### Python Style

- Follow PEP 8
- Use type hints where possible
- Max line length: 100 characters
- Use docstrings (Google style)

Example:
```python
def generate_code(self, spec: MMMSpec) -> str:
    """
    Generate MMM code from specification.
    
    Args:
        spec: The MMM specification
        
    Returns:
        Generated Python code as string
        
    Raises:
        ValueError: If spec is invalid
    """
    pass
```

### Testing

- Write tests for new features
- Maintain >80% code coverage
- Use pytest fixtures for common setups

Example:
```python
def test_spec_generation():
    """Test that spec generation works correctly"""
    writer = SpecWriterAgent(llm)
    spec = writer.generate_spec(data_path="test_data.csv")
    assert spec.outcome is not None
    assert len(spec.channels) > 0
```

## 🎯 Areas for Contribution

### High Priority

1. **Agent Improvements**
   - Better error handling
   - More robust code generation
   - Support for more model types

2. **RAG Enhancements**
   - Add more production examples
   - Better retrieval strategies
   - Domain-specific embeddings

3. **Testing**
   - Integration tests
   - Edge case coverage
   - Performance benchmarks

4. **Documentation**
   - Tutorial notebooks
   - Video walkthroughs
   - API documentation

### Medium Priority

1. **New Features**
   - Hierarchical models
   - Experiment calibration
   - Interactive UI (Streamlit)

2. **Infrastructure**
   - Docker deployment
   - Cloud templates (AWS, GCP, Azure)
   - MLflow integration

3. **Performance**
   - Faster code generation
   - Caching strategies
   - Parallel agent execution

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Reproduction**: Minimal code to reproduce
3. **Expected**: What you expected to happen
4. **Actual**: What actually happened
5. **Environment**: 
   - OS (macOS, Linux, Windows)
   - Python version
   - Cerebro version
   - LLM backend (Ollama, API)

Example:
```markdown
**Bug**: SpecWriterAgent fails on CSV with special characters

**Reproduction**:
```python
writer = SpecWriterAgent(llm)
spec = writer.generate_spec("data_with_émojis.csv")
```

**Expected**: Generates valid spec
**Actual**: UnicodeDecodeError

**Environment**:
- macOS 14.0
- Python 3.9
- Cerebro 0.1.0
- Ollama (Qwen 7B)
```

## 💡 Feature Requests

When suggesting features:

1. **Use Case**: Describe the problem/use case
2. **Proposed Solution**: Your suggested approach
3. **Alternatives**: Other options considered
4. **Examples**: Mock code or screenshots

## 🔍 Code Review Process

All contributions go through code review:

1. **Automated Checks**: CI runs tests and linting
2. **Maintainer Review**: A maintainer reviews code
3. **Feedback**: Address any requested changes
4. **Merge**: Once approved, PR is merged

Review criteria:
- ✅ Tests pass
- ✅ Code follows style guidelines
- ✅ Documentation is updated
- ✅ No breaking changes (or documented)

## 📚 Resources

- **Documentation**: [README.md](README.md)
- **Examples**: [examples/](examples/)
- **Architecture**: See agent source code in `cerebro/agents/`
- **RAG System**: [fine_tuning/](fine_tuning/)

## 🤝 Community

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas
- **Discord** (coming soon): Real-time chat

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Cerebro! 🧠✨

