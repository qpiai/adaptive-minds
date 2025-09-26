# Contributing to Adaptive Minds

Thank you for your interest in contributing to Adaptive Minds! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions of all kinds:

- 🐛 **Bug reports** and fixes
- ✨ **New features** and enhancements
- 📚 **Documentation** improvements
- 🧠 **New domain adapters** and models
- 🧪 **Tests** and benchmarks
- 💡 **Ideas** and suggestions

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker with GPU support
- NVIDIA GPU with 8GB+ VRAM
- Git
- Hugging Face account

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/adaptive_minds_oss.git
   cd adaptive_minds_oss
   ```

2. **Set up your environment:**
   ```bash
   # Set your Hugging Face token
   export HF_TOKEN=your_token_here
   
   # Install dependencies
   pip install -r build/requirements.txt
   
   # Download models for testing
   python build/download_models.py
   ```

3. **Run the system locally:**
   ```bash
   # Terminal 1: Start the server
   cd build && python server.py
   
   # Terminal 2: Start the frontend
   cd build && streamlit run app_frontend.py
   ```

4. **Test your setup:**
   ```bash
   curl -X POST http://localhost:8765/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, how are you?"}'
   ```

## 📋 Contribution Types

### 🐛 Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to reproduce**: Minimal steps to reproduce the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, GPU model, Docker version
- **Logs**: Relevant error messages or logs

**Use the bug report template when creating issues.**

### ✨ Feature Requests

For new features:

- **Description**: Clear description of the proposed feature
- **Use case**: Why is this feature needed?
- **Implementation ideas**: Any thoughts on how to implement it
- **Examples**: Mockups, code snippets, or examples

### 🧠 Adding New Domain Adapters

We encourage adding new domain-specific adapters! Here's how:

1. **Train your LoRA adapter** using your domain-specific data
2. **Upload to Hugging Face** with proper model cards
3. **Add to the system** by updating the configuration
4. **Test thoroughly** with domain-specific queries
5. **Document** the adapter's capabilities and limitations

#### Adapter Requirements

- **Model base**: Must be compatible with Llama 3.1 8B
- **Format**: LoRA adapter using PEFT
- **Documentation**: Clear model card on Hugging Face
- **Testing**: Include test queries and expected behaviors
- **Licensing**: Must be compatible with Apache 2.0

#### Example Adapter Addition

```python
# In server.py, add to LORA_ADAPTERS
"Legal": {
    "path": "/app/loras/llama-8B-legal",
    "description": "Legal expert specializing in contracts, regulations, and legal advice",
    "system_prompt": "You are a legal expert. Provide accurate legal information while always recommending consultation with qualified attorneys for specific legal matters.",
    "keywords": "legal, law, contract, regulation, attorney, court, lawsuit, compliance"
}
```

## 🔧 Development Guidelines

### Code Style

- **Python**: Follow PEP 8 style guidelines
- **Formatting**: Use `black` for code formatting
- **Linting**: Use `flake8` for linting
- **Type hints**: Include type hints where appropriate
- **Docstrings**: Use Google-style docstrings

```bash
# Format code
black build/

# Lint code
flake8 build/

# Type checking
mypy build/
```

### Commit Messages

Use conventional commit messages:

```
feat: add new chemistry adapter
fix: resolve GPU memory leak in model loading
docs: update API documentation
test: add routing accuracy tests
refactor: simplify adapter configuration
```

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding guidelines

3. **Test your changes:**
   ```bash
   # Run the system
   docker compose up --build
   
   # Test API endpoints
   curl -X POST http://localhost:8765/chat -H "Content-Type: application/json" -d '{"query": "test query"}'
   
   # Test web interface
   open http://localhost:8501
   ```

4. **Update documentation** if needed

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```

6. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request** with:
   - Clear title and description
   - Reference any related issues
   - Include screenshots/demos if applicable
   - Ensure CI passes

### Testing

- **Unit tests**: Test individual components
- **Integration tests**: Test the full workflow
- **Performance tests**: Ensure no regressions
- **Manual testing**: Test the UI and API endpoints

```bash
# Run tests (when available)
pytest tests/

# Test routing accuracy
python tests/test_routing_accuracy.py

# Performance benchmarks
python benchmarks/response_time.py
```

## 📁 Project Structure

```
adaptive_minds_oss/
├── build/                     # Main application code
│   ├── server.py             # FastAPI backend
│   ├── app_frontend.py       # Streamlit frontend
│   ├── app_standalone.py     # Standalone Streamlit app
│   ├── download_models.py    # Model downloader
│   └── requirements.txt      # Python dependencies
├── tests/                    # Test files (to be added)
├── docs/                     # Additional documentation (to be added)
├── examples/                 # Usage examples (to be added)
├── benchmarks/               # Performance benchmarks (to be added)
├── docker-compose.yml        # Docker deployment
├── Dockerfile               # Container definition
├── README.md                # Main documentation
├── CONTRIBUTING.md          # This file
├── LICENSE                  # Apache 2.0 License
└── .gitignore              # Git exclusions
```

## 🎯 Priority Areas for Contribution

We're particularly looking for help with:

1. **🧪 Testing Framework**
   - Unit tests for routing logic
   - Integration tests for the full system
   - Performance benchmarks
   - Routing accuracy evaluation

2. **🧠 New Domain Adapters**
   - Legal domain
   - Education/tutoring
   - Creative writing
   - Technical documentation
   - Customer support

3. **🔧 Developer Tools**
   - CLI for easier setup and management
   - Configuration system (YAML-based)
   - Model evaluation tools
   - Performance monitoring

4. **📚 Documentation**
   - API documentation
   - Deployment guides
   - Tutorial videos
   - Use case examples

5. **🎨 User Interface**
   - Gradio alternative frontend
   - React/Next.js web app
   - Mobile-responsive design
   - Accessibility improvements

## 🏷️ Issue Labels

We use these labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `adapter`: Related to domain adapters
- `performance`: Performance improvements
- `ui/ux`: User interface improvements
- `testing`: Testing related

## 💬 Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Be patient** with newcomers
- **Be professional** in all interactions

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📄 License

By contributing to Adaptive Minds, you agree that your contributions will be licensed under the Apache 2.0 License.

## ❓ Questions?

If you have questions about contributing, please:

1. Check existing [GitHub Issues](https://github.com/yourusername/adaptive_minds_oss/issues)
2. Create a new issue with the `question` label
3. Start a [GitHub Discussion](https://github.com/yourusername/adaptive_minds_oss/discussions)

Thank you for contributing to Adaptive Minds! 🚀
