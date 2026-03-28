# Contributing to ToolOrchestra

Thank you for your interest in contributing to ToolOrchestra! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ToolOrchestra.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Submit a pull request

## Development Environment

See the [README.md](README.md) for detailed setup instructions. Key requirements:

- Python 3.12
- CUDA 12.4+
- NVIDIA GPU (Ampere or newer recommended)

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings for public functions and classes
- Avoid bare `except:` clauses - use specific exception types

## Testing

Before submitting a PR:
- Test your changes locally
- Ensure existing functionality still works
- Add tests for new features if applicable

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Ensure your PR description clearly describes the problem and solution
3. Link any related issues using keywords (e.g., "Fixes #123")
4. Wait for review from maintainers

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce
- Your environment details (OS, Python version, CUDA version)
- Error messages or logs

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
