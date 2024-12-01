Sentinel Leaf Classification
==============================

> Disclaimer: This repository is a work in progress and the corresponding paper has not been published yet. The code is subject to change.

Welcome to the repository for the paper "Evergreen Leaf Classification Using Tailored Sentinel-2 Composites." Our goal is to ensure all results are reproducible. If you encounter any problems, please open an issue in this repository.

# Overview

In this study, we investigate the potential of Sentinel-2 Level-2A data for classifying evergreen leafs of forests in temperate climates. We focus on optimizing spectral band combinations, compositing methods, and hyperparameters for various classification models to enhance Sentinel-2 Level-2A data for classifying evergreen leafs. The experiments are broken down into:

1. **Band Importance:** We conduct a systematic evaluation using recursive feature elimination (RFE) with a random forest model to identify significant bands and indices.
2. **Compositing:** The best number of composites per year is determined for each compositing method to refine the dataset.
3. **Hyperparameter Tuning:** Hyperparameters for multiple classification models are optimized and the best model is selected based on predictive quality.
4. **Generalization:** We evaluate the best model's generalization capability across one large coherent unseen experimental site.

Our results highlight the potential of tailored Sentinel-2 composites for classifying evergreen leafs and provide insights into optimizing spectral band combinations and compositing methods.

# Installation and Setup

This repository requires **Python 3.10** or later. To install the required packages we recommend using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Install all necessary packages by running the following commands in the root directory of this repository:

```bash
mamba create -n slc -y xgboost uv
mamba activate slc
uv pip install -e '.[test,lint]'
```

To use the **Earth Engine API**, authenticate with `python -c "import ee;ee.Authenticate()"`. A browser tab should open after you execute the code. After following the website instructions you can use the API. Repeat this process whenever your token expires.

# Usage

> **Note:** The models and studies are serialized from PKL files. To prevent arbitrary code execution, **never** load models from untrusted sources.

All experiments described in our paper are implemented in the notebooks within the `notebooks` directory. After completing the installation and setup, you can run these notebooks in your preferred IDE.

# CI Pipeline

This GitHub repository uses [GitHub Actions](https://github.com/features/actions) for continuous integration (CI). The CI pipeline is defined in the `.github/workflows` directory. It includes:
- **Linting** with [ruff](https://docs.astral.sh/ruff/)
- **Testing** with [pytest](https://docs.pytest.org/)

To run the CI workflow locally, execute the following commands from the repository directory:

```bash
ruff check --select I --fix
ruff format
pytest
```

The code was formatted automatically using following VS Code settings alongside the `charliermarsh.ruff` extension:

```json
"notebook.formatOnSave.enabled": true,
"notebook.codeActionsOnSave": {
    "notebook.source.fixAll": "explicit",
    "notebook.source.organizeImports": "explicit",
},
"python.analysis.ignore": ["*"],
"[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": "always",
        "source.organizeImports": "always",
    },
    "editor.defaultFormatter": "charliermarsh.ruff",
},
"files.autoSave": "onFocusChange",
```

# Acknowledgements

We would like to thank all the contributors and reviewers for their valuable feedback and support.

# Contact

For questions or contributions, please contact Peter Hofinger at [hofinger-peter@gmx.de](hofinger-peter@gmx.de).

# Citation

If you use this repository in your research, please cite our paper:

```
@article{paper2024,
  title={Paper Title},
  author={Name and Coauthor Names},
  journal={Journal Name},
  year={2024},
  volume={xx},
  number={xx},
  pages={xx--xx},
  doi={xx.xxxx/xxxxxx},
}
```

Thank you for using our code and contributing to the advancement of remote sensing!
