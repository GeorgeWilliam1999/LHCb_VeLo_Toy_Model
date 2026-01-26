# Contributing to LHCb VELO Toy Model

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LHCb_VeLo_Toy_Model.git
   cd LHCb_VeLo_Toy_Model
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/GeorgeWilliam1999/LHCb_VeLo_Toy_Model.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Optional Dependencies

For quantum algorithm development:
```bash
pip install -e ".[quantum]"
```

For Jupyter notebook work:
```bash
pip install -e ".[notebook]"
```

For documentation:
```bash
pip install -e ".[docs]"
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-metric` - for new features
- `fix/ghost-rate-calculation` - for bug fixes
- `docs/update-readme` - for documentation changes
- `refactor/hamiltonian-solver` - for code refactoring

### Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Write tests** for any new functionality

4. **Run tests** to ensure nothing is broken:
   ```bash
   pytest
   ```

5. **Commit your changes** with clear messages:
   ```bash
   git add .
   git commit -m "Add feature: description of the change"
   ```

## Code Style

We use several tools to maintain code quality:

### Formatting

- **Black** for code formatting (line length: 88)
- **isort** for import sorting

Run formatters:
```bash
black src/ tests/
isort src/ tests/
```

### Linting

- **Ruff** for fast Python linting
- **mypy** for type checking

Run linters:
```bash
ruff check src/ tests/
mypy src/
```

### Style Guidelines

1. **Type hints**: Add type hints to function signatures
   ```python
   def calculate_efficiency(tracks: list[Track], truth: list[Track]) -> float:
       ...
   ```

2. **Docstrings**: Use NumPy-style docstrings
   ```python
   def solve_hamiltonian(A: np.ndarray, b: np.ndarray) -> np.ndarray:
       """
       Solve the Hamiltonian system Ax = b.

       Parameters
       ----------
       A : np.ndarray
           The Hamiltonian matrix.
       b : np.ndarray
           The bias vector.

       Returns
       -------
       np.ndarray
           The solution vector.

       Examples
       --------
       >>> A = np.array([[1, 0], [0, 1]])
       >>> b = np.array([1, 2])
       >>> solve_hamiltonian(A, b)
       array([1., 2.])
       """
       ...
   ```

3. **Imports**: Group imports in order:
   - Standard library
   - Third-party packages
   - Local imports

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=velo_toy --cov-report=html

# Run specific test file
pytest tests/test_hamiltonian.py

# Run tests matching a pattern
pytest -k "test_efficiency"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_hamiltonian_returns_sparse_matrix`
- Use fixtures for common setup
- Test edge cases and error conditions

Example test:
```python
import pytest
import numpy as np
from velo_toy.core import SimpleHamiltonian

class TestSimpleHamiltonian:
    def test_construct_hamiltonian_returns_sparse(self):
        ham = SimpleHamiltonian(epsilon=0.001, gamma=1.0, delta=1.0)
        # ... test implementation
        
    def test_solve_classically_returns_array(self):
        # ... test implementation
        
    @pytest.mark.parametrize("epsilon", [0.0001, 0.001, 0.01])
    def test_different_epsilon_values(self, epsilon):
        # ... test implementation
```

## Submitting Changes

### Pull Request Process

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe the changes made
   - Include any relevant screenshots or results

3. **Address review feedback**:
   - Make requested changes
   - Push updates to the same branch
   - Re-request review when ready

### Pull Request Checklist

- [ ] Code follows the style guidelines
- [ ] Tests pass locally (`pytest`)
- [ ] New functionality includes tests
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with `main`

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, package versions

### Feature Requests

For feature requests, please include:

1. **Description**: Clear description of the feature
2. **Use case**: Why this feature would be useful
3. **Proposed solution**: If you have ideas on implementation

## Questions?

If you have questions, feel free to:
- Open an issue on GitHub
- Contact the maintainers

Thank you for contributing!
