# MyDiffusion

A PyTorch implementation of diffusion models for scientific computing, specifically designed for plasma physics data analysis (CTS - Coherent Thomson Scattering).

## Features

- **Unconditional and Conditional Diffusion Models**: Support for both standard and conditional generation
- **Variance Exploding SDE (VE-SDE)**: Advanced noise scheduling for better sample quality
- **Likelihood-Guided Sampling**: Integration with physical constraints and measurements
- **Classifier-Free Guidance**: Improved conditional generation without separate classifiers
- **Scientific Data Focus**: Designed for plasma physics and other scientific applications

## Installation

```bash
# Install in development mode (recommended)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import diffusion
import yaml

# Load configuration
config = yaml.safe_load(open("model_config.yaml"))

# Create and train model
model = diffusion.Main(config)
model.train(dataset, save_path='my_model')

# Generate samples
samples, trajectory = model.sampling(nsamples=100)
```

## Project Structure

```
src/diffusion/
├── __init__.py         # Package imports
├── sde.py             # Main training/sampling logic
├── nn_model.py        # Neural network architecture
├── dataset.py         # Data handling abstractions
└── likelihood.py      # Likelihood and operators

tests/Dataset/         # Training examples and datasets
├── Training_example.ipynb
├── Eval_example.ipynb
└── cts_training.csv
```

## Dependencies

- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0
- PyYAML >= 6.0

## License

MIT License