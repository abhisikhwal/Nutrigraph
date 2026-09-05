# Contributing to NutriGraph

Thank you for your interest in contributing to NutriGraph (repository: `global-food-genome`).

## Getting Started

### Prerequisites
- Python 3.11+
- Conda (recommended) or pip
- Git

### Setup Development Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd global-food-genome
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate food-genome
```

3. Install in development mode:
```bash
pip install -e .
```

4. Verify installation:
```bash
python -c "import src; print('Success!')"
pytest tests/
```

---

## Code Style

### Python Style Guide
- Follow PEP 8
- Use type hints where applicable
- Maximum line length: 88 characters (Black default)
- Docstrings: Google style

### Formatting Tools
We use automated formatting:

```bash
# Format code
black src/ scripts/ tests/

# Check style
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

### Docstring Example
```python
def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index.
    
    Args:
        red_band: Red band reflectance values
        nir_band: Near-infrared reflectance values
        
    Returns:
        NDVI values in range [-1, 1]
        
    Example:
        >>> ndvi = calculate_ndvi(red, nir)
        >>> print(ndvi.mean())
        0.75
    """
```

---

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

### 2. Make Changes
- Write code following style guidelines
- Add tests for new functionality
- Update documentation

### 3. Test Your Changes
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data/test_validators.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### 4. Commit Changes
Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add origin feature extraction for satellite data

- Implement NDVI and EVI calculation
- Add temporal aggregation utilities
- Include tests for spatial feature extraction
"
```

Commit message format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description with bullet points

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub/GitLab.

---

## Testing Guidelines

### Writing Tests

1. **Location**: Place tests in `tests/test_<module>/`
2. **Naming**: Test files must start with `test_`
3. **Structure**: Use fixtures for reusable test data

Example:
```python
import pytest
from src.data.validators import SchemaValidator

@pytest.fixture
def validator():
    return SchemaValidator()

def test_validate_valid_dataframe(validator):
    df = create_sample_dataframe()
    result = validator.validate_dataframe(df, 'ingredient_schema')
    assert result['valid'] is True
```

### Test Categories

Mark tests appropriately:
```python
@pytest.mark.slow
def test_large_dataset():
    pass

@pytest.mark.requires_rdkit
def test_fingerprint_generation():
    pass
```

### Running Tests

```bash
# All tests
pytest

# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m integration

# With coverage
pytest --cov=src --cov-report=term
```

---

## Adding New Datasets

When adding support for a new dataset:

1. **Update license registry**:
   - Add entry to `licenses/datasets_registry.csv`
   - Document license in `licenses/compliance_notes.md`

2. **Add dataset configuration**:
   - Add metadata to `config/datasets.yaml`
   - Add paths to `config/paths.yaml`

3. **Create downloader**:
   - Add download script in appropriate phase folder
   - Use `DatasetDownloader` base class

4. **Create parser**:
   - Add parser to `src/data/parsers.py`
   - Follow existing patterns

5. **Add schema validation**:
   - Define JSON schema if new data type
   - Validate output in processing script

6. **Update documentation**:
   - Add to `docs/dataset_sources.md`
   - Update README if necessary

---

## Documentation

### Code Documentation
- All public functions must have docstrings
- Include type hints
- Provide examples where helpful

### Project Documentation
Located in `docs/`:
- `architecture.md` - System design
- `dataset_sources.md` - Data provenance
- `api_reference.md` - API documentation

### Notebooks
Demonstration notebooks in `notebooks/demo/`:
- Clear markdown explanations
- Reproducible examples
- Expected outputs shown

---

## Dataset License Compliance

**CRITICAL**: Before adding any dataset:

1. Check license compatibility
2. Update `licenses/datasets_registry.csv`
3. Flag non-commercial restrictions
4. Document in `licenses/compliance_notes.md`

Never commit:
- Raw datasets (add to `.gitignore`)
- API keys or credentials
- Proprietary data

---

## Code Review Process

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass (`pytest`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] License compliance checked (if adding data)
- [ ] Commit messages are clear
- [ ] No sensitive data in commits

### Review Criteria

Reviewers will check:
1. Correctness and functionality
2. Code quality and readability
3. Test coverage
4. Documentation completeness
5. License compliance

---

## Questions?

For questions or discussions:
- Open an issue on GitHub
- Contact project maintainers
- Check existing documentation

---

Thank you for contributing to NutriGraph!
