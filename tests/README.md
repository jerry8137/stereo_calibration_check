# Tests

This directory contains automated tests for the stereo calibration check package.

## Setup

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Or with uv:

```bash
uv pip install -e ".[dev]"
```

## Running Tests

Run all tests:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=stereo_calibration_check --cov-report=html
```

Run specific test file:

```bash
pytest tests/test_baseline.py
```

Run specific test:

```bash
pytest tests/test_baseline.py::TestBaselineParameter::test_default_baseline
```

Run tests matching a pattern:

```bash
pytest -k "baseline"
```

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_baseline.py` - Unit tests for baseline parameter functionality
- `test_cli.py` - Integration tests for CLI interface

## Writing New Tests

When adding new tests:

1. Place unit tests in appropriate test files
2. Use fixtures from `conftest.py` for common setup
3. Follow the naming convention: `test_*.py` for files, `test_*` for functions
4. Use descriptive test names that explain what is being tested
