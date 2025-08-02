# Contributing to Adaptive Fuzzy-PSO DBSCAN

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Adaptive Fuzzy-PSO DBSCAN implementation.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/adaptive-fuzzy-pso-dbscan.git
   cd adaptive-fuzzy-pso-dbscan
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep line length under 88 characters
- Use type hints where possible

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Test both normal and edge cases
- Include integration tests for major features

## Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add appropriate tests
   - Update documentation if needed

3. **Run Quality Checks**
   ```bash
   # Linting
   flake8 .
   
   # Type checking
   mypy --ignore-missing-imports .
   
   # Tests
   pytest tests/ -v
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use descriptive title and description
   - Reference any related issues
   - Include screenshots for UI changes

## Commit Message Format

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `style:` Code style changes

## Areas for Contribution

### Algorithm Improvements
- PSO parameter optimization enhancements
- Fuzzy membership function variations
- Alternative clustering metrics
- Performance optimizations

### Visualization Features
- Additional chart types
- Interactive parameter tuning
- Real-time clustering updates
- Export formats

### Data Processing
- Support for additional data formats
- Advanced preprocessing options
- Data validation improvements
- Error handling enhancements

### Documentation
- API documentation
- Tutorial notebooks
- Performance benchmarks
- Use case examples

## Bug Reports

When reporting bugs, include:
- Operating system and Python version
- Complete error traceback
- Steps to reproduce
- Expected vs actual behavior
- Sample data (if applicable)

## Feature Requests

For new features, provide:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant research or references

## Code Review Guidelines

### For Reviewers
- Check code quality and style
- Verify test coverage
- Ensure documentation is updated
- Test functionality locally
- Provide constructive feedback

### For Contributors
- Respond to review comments promptly
- Make requested changes
- Update PR description if scope changes
- Rebase if necessary

## Research Contributions

This project implements research by Seid Mehammed Abdu and Md Nasre Alam. When contributing:

- Maintain scientific accuracy
- Cite relevant literature
- Document algorithmic changes
- Validate against research benchmarks

## Community Guidelines

- Be respectful and inclusive
- Help newcomers get started
- Share knowledge and expertise
- Follow the code of conduct

## Getting Help

- Check existing issues and documentation
- Join discussions in GitHub issues
- Contact maintainers for complex questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to advancing smart city analytics and clustering research!