# genjax-examples

Examples and tutorials for GenJax, a probabilistic programming library built on JAX.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/genjax-examples.git
cd genjax-examples
```

2. Create virtual environment and install dependencies:

```bash
uv venv
```

This will create a virtual environment at: `.venv`
Activate with: `source .venv/bin/activate`

```bash
uv pip install -e .
```

3. Run examples with our virtual environment activated

```bash
python tests/test_changepoint_model.py
```

## Usage

Browse the `examples/` directory for various GenJax tutorials and demonstrations.
