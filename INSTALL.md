# Installation Guide for CeDNe

This guide provides step-by-step instructions for installing and setting up CeDNe, a Python-based framework for modeling and analyzing the nervous system.

---

## System Requirements

| Component      | Version / Info                   |
|----------------|----------------------------------|
| Python         | ≥ 3.9 (≤ 3.12 recommended)       |
| OS             | macOS, Linux, or WSL2 (Windows)  |
| Disk Space     | ~500 MB (after virtual env setup)|
| RAM            | 2 GB minimum (4 GB recommended)  |

---

## Poetry Setup

CeDNe uses [Poetry](https://python-poetry.org/) for environment and dependency management.

### Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After installation, add Poetry to your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

You can add this to your shell config (`.bashrc`, `.zshrc`, etc.) for persistence.

> **Check**: `poetry --version` should return something like `Poetry (1.7.1)`

---

## Install CeDNe

```bash
# Clone the repository
git clone https://github.com/sahilm89/CeDNe.git
cd CeDNe

# Install dependencies and create the virtual environment
poetry install
```

Activate the environment:

```bash
poetry shell
```

---

## Jupyter Notebooks & Visualization (Optional)

Some features require additional tools like `jupyter`, `matplotlib`, and `plotly`.

```bash
poetry add notebook matplotlib plotly ipywidgets --group dev
```

To run the example notebooks:

```bash
jupyter notebook
```

> All example notebooks are located in the [`examples/notebooks`](examples/notebooks) directory.

---

## Troubleshooting

| Problem                             | Fix |
|------------------------------------|-----|
| `poetry: command not found`        | Ensure `$HOME/.local/bin` is in your `PATH` |
| Python version incompatible        | Use `pyenv` or `conda` to install Python 3.9+ |
| Notebook fails to visualize plots  | Ensure `matplotlib`, `plotly`, and Jupyter are installed inside Poetry env |

---

## Updating CeDNe

```bash
git pull
poetry install
```

To update all dependencies:

```bash
poetry update
```