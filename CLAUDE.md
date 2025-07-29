# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python diffusion model implementation for scientific computing, specifically designed for plasma physics data analysis (CTS - Coherent Thomson Scattering). The project implements denoising diffusion probabilistic models using PyTorch.

## Important Development Guidelines

### Git Repository
- This project is now a Git repository
- Always commit changes with descriptive messages
- Use conventional commit format when possible

### Code Quality
- Run linting and formatting before committing:
  ```bash
  # With Poetry
  poetry run black src/
  poetry run isort src/
  poetry run flake8 src/
  
  # With pip
  black src/
  isort src/
  flake8 src/
  ```

## Architecture

### Core Components

- **`src/diffusion/sde.py`**: Main training and sampling orchestration
  - `Main` class: Handles unconditional diffusion model training and sampling
  - `Main_conditional` class: Handles conditional diffusion model training with classifier-free guidance
  - `Sigma` class: Manages noise scheduling (variance exploding SDE)
  - Conditional sampling functions with likelihood-based guidance

- **`src/diffusion/nn_model.py`**: Neural network architecture
  - `DiffusionModel`: Main score network (time-conditioned MLP)
  - `DiffusionBlock`: Building blocks with optional batch normalization
  - Supports conditional generation with condition concatenation

- **`src/diffusion/dataset.py`**: Data handling abstractions
  - `Mydataset`: Abstract base class for unconditional datasets
  - `Mydataset_conditional`: Abstract base class for conditional datasets
  - Built-in normalization and DataLoader integration

- **`src/diffusion/likelihood.py`**: Likelihood and operator abstractions
  - `Likelihood`: Abstract base for likelihood-guided sampling
  - `LinearOperator`: Abstract base for forward/inverse problems
  - Registration system for different operators

### Training Process

The implementation uses variance exploding SDE (VE-SDE) with:
- MSE loss between predicted and true noise
- AdamW optimizer with ReduceLROnPlateau scheduling
- Gradient clipping (max_norm=1.0)
- Checkpoint saving every N epochs

### Sampling Methods

- Standard DDPM-style sampling
- Conditional sampling with likelihood guidance
- Classifier-free guidance for conditional models
- Support for Gaussian process priors in sampling

## Installation and Development

The package supports both pip and Poetry for dependency management:

### Option 1: Using pip (Simple)

```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install normally
pip install .

# Install with development dependencies
pip install -e ".[dev]"
```

### Option 2: Using Poetry (Recommended for isolated environments)

```bash
# Install Poetry (if not already installed)
pip install poetry

# Install dependencies and create virtual environment
poetry install

# Activate Poetry shell
source $(poetry env info --path)/bin/activate

# Or run commands in Poetry environment
poetry run python your_script.py
poetry run jupyter notebook
```

### Development Commands

#### With pip:
```bash
# Install the package in development mode
pip install -e .

# Run training examples (after installation)
cd tests/Dataset/
jupyter notebook Training_example.ipynb

# Run linting and formatting (if dev dependencies installed)
black src/
isort src/
flake8 src/

# Run tests (if pytest and tests exist)
pytest
```

#### With Poetry:
```bash
# Install dependencies
poetry install

# Run training examples
cd tests/Dataset/
poetry run jupyter notebook Training_example.ipynb

# Run linting and formatting
poetry run black src/
poetry run isort src/
poetry run flake8 src/

# Run tests
poetry run pytest

# Add new dependencies
poetry add new-package
poetry add --group dev new-dev-package
```

### CUDA Support

Both installation methods support CUDA:
- Poetry automatically installs PyTorch with CUDA support
- The package will work with GPU acceleration when CUDA is available
- Virtual environment path: `~/.cache/pypoetry/virtualenvs/mydiffusion-*/`

## Configuration

Models are configured via YAML files (see `tests/Dataset/Training_example.ipynb`):
- `nn_model`: Architecture parameters (nfeatures, nblocks, nunits, batch_norm)
- `train`: Training parameters (nepochs, batch_size, lr, optimizer settings)
- `sampler`: Sampling parameters (nstep, sigma_min, sigma_max)

## Key Implementation Details

- Uses time-conditioning by concatenating normalized timesteps to input
- Supports both unconditional and conditional generation
- Implements classifier-free guidance with null condition training
- Includes sophisticated likelihood-guided sampling for inverse problems
- Built-in data normalization and denormalization utilities

## File Structure

```
MyDiffusion/
├── CLAUDE.md                    # This file - project guidance
├── COMMANDS.md                  # Command reference
├── POETRY_GUIDE.md             # Poetry usage guide
├── README.md                   # Project documentation
├── pyproject.toml              # Poetry configuration
├── poetry.lock                 # Locked dependencies
├── src/diffusion/
│   ├── __init__.py             # Package imports
│   ├── sde.py                  # Main training/sampling logic
│   ├── nn_model.py             # Neural network architecture
│   ├── dataset.py              # Data handling abstractions
│   └── likelihood.py           # Likelihood and operators
└── tests/
    ├── Training_example.ipynb   # Training tutorial
    ├── Eval_example.ipynb      # Evaluation tutorial
    ├── create_training_data.py # Data generation script
    ├── cts.py                  # CTS-specific implementation
    ├── model_config.yaml       # Model configuration
    ├── edm_setting.yaml        # EDM settings
    └── Dataset/
        └── cts_training.csv    # Training data
```

## Additional Documentation

- **COMMANDS.md**: Quick reference for common commands
- **POETRY_GUIDE.md**: Detailed Poetry usage instructions
- **README.md**: Main project documentation and setup guide

The codebase follows a clean separation between data handling, model architecture, and training orchestration, making it easy to extend for different scientific applications.