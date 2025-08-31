# Iris Documentation

Welcome to the Iris documentation! This directory contains comprehensive documentation for the Iris multi-GPU programming framework, built using Sphinx and following the Diataxis framework.

## 📚 Documentation Structure

The documentation is organized following the [Diataxis](https://diataxis.fr/) framework for effective learning:

### **Getting Started** 🚀
- **[Installation Guide](getting-started/installation.md)**: Set up Iris on your system
- **[Quick Start Guide](getting-started/quick-start.md)**: Run your first multi-GPU program

### **Tutorials** 🎯
- **[Examples](reference/examples.md)**: Complete working examples for all operations
- **[API Reference](reference/iris-api.md)**: Auto-generated API documentation

### **Examples & API** 🔧
- **[Examples](reference/examples.md)**: Complete working examples for all operations
- **[API Reference](reference/iris-api.md)**: Auto-generated API documentation

### **Conceptual** 🧠
- **[Programming Model](conceptual/programming-model.md)**: Deep dive into how Iris works
- **[Fine-grained Overlap](conceptual/finegrained-overlap.md)**: Advanced optimization techniques
- **[Architecture](conceptual/architecture.md)**: System design and internals

### **Reference** 📖
- **[API Reference](reference/iris-api.md)**: Complete API documentation
- **[Examples](reference/examples.md)**: Comprehensive example collection
- **[Contributing](reference/contributing.md)**: How to contribute to Iris

## 🛠️ Building the Documentation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Quick Build

Use the provided build script:

```bash
cd docs
./build_docs.sh
```

### Manual Build

1. **Create virtual environment:**
   ```bash
   cd docs
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r sphinx/requirements.txt
   ```

3. **Build documentation:**
   ```bash
   python3 -m sphinx -b html -d _build/doctrees -D language=en . _build/html
   ```

4. **View documentation:**
   ```bash
   # Option 1: Open in browser
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

   # Option 2: Serve locally
   python3 -m http.server -d _build/html/
   # Then visit http://localhost:8000
   ```

### Auto-build (Development)

For development with automatic rebuilding:

```bash
# Install autobuild
pip install sphinx-autobuild

# Run autobuild
sphinx-autobuild -b html -d _build/doctrees -D language=en . _build/html --ignore "_build/*" --ignore "sphinx/_toc.yml" --ignore "sphinx/requirements.txt"
```

## 📖 Reading the Documentation

### For Beginners

1. **Start with**: [Installation Guide](getting-started/installation.md)
2. **Then**: [Quick Start Guide](getting-started/quick-start.md)
3. **Practice**: [Examples](reference/examples.md)

### For Intermediate Users

1. **Study**: [Programming Model](conceptual/programming-model.md)
2. **Explore**: [Examples](reference/examples.md)
3. **Reference**: [API Reference](reference/iris-api.md)
4. **Contribute**: [Contributing Guide](reference/contributing.md)

### For Advanced Users

1. **Master**: [Fine-grained Overlap](conceptual/finegrained-overlap.md)
2. **Reference**: [API Reference](reference/iris-api.md)
3. **Contribute**: [Contributing Guide](reference/contributing.md)
4. **Extend**: [Architecture](conceptual/architecture.md)

## 🔧 Configuration

### Sphinx Configuration

The main Sphinx configuration is in `conf.py`:

- **Theme**: Read the Docs theme
- **Extensions**: autodoc, napoleon, viewcode, intersphinx, myst_parser
- **Language**: English
- **Output**: HTML

### Table of Contents

Navigation structure is defined in `sphinx/_toc.yml.in`:

```yaml
root: index.md
subtrees:
  - caption: Getting Started
    entries:
    - file: getting-started/installation.md
    # ... more entries
```

### Requirements

Documentation dependencies are in `sphinx/requirements.txt`:

- sphinx>=5.0.0
- sphinx-rtd-theme>=1.0.0
- myst-parser>=1.0.0
- sphinx-copybutton>=0.5.0

## 📝 Contributing to Documentation

### Adding New Pages

1. **Create file** in appropriate directory
2. **Update TOC** in `sphinx/_toc.yml.in`
3. **Add links** from related pages
4. **Test build** locally

### Documentation Standards

1. **Use Markdown**: All pages use MyST Markdown
2. **Follow Diataxis**: Organize by purpose (tutorials, how-to, conceptual, reference)
3. **Include examples**: Provide working code examples
4. **Cross-reference**: Link to related documentation

### Style Guide

1. **Headers**: Use descriptive, hierarchical headers
2. **Code blocks**: Include syntax highlighting and explanations
3. **Links**: Use relative links within documentation
4. **Images**: Include alt text and captions

## 🚀 Deployment

### ReadTheDocs

The documentation is configured for ReadTheDocs deployment:

- **Configuration**: `.readthedocs.yaml` in repository root
- **Build**: Automatic builds on push to main branch
- **Domain**: Available at readthedocs.io

### Local Deployment

For local or internal deployment:

```bash
# Build static files
./build_docs.sh

# Deploy to web server
cp -r _build/html/* /var/www/iris-docs/
```

## 📚 Additional Resources

### External Links

- **[Iris Repository](https://github.com/ROCm/iris)**: Source code and issues
- **[GitHub Discussions](https://github.com/ROCm/iris/discussions)**: Community support
- **[ROCm Documentation](https://rocmdocs.amd.com/)**: AMD GPU computing platform

### Related Documentation

- **[Triton Documentation](https://triton-lang.org/main/)**: GPU programming language
- **[PyTorch Documentation](https://pytorch.org/docs/)**: Deep learning framework
- **[MPI Documentation](https://www.mpi-forum.org/docs/)**: Message passing interface

---

**Ready to start learning Iris? Begin with the [Installation Guide](getting-started/installation.md)!**

*This documentation is maintained by the Iris development team and community contributors.*
