# TAFA: Template-Assisted Feature Acquisition

This repository contains the implementation of **Template-Assisted Feature Acquisition (TAFA)**, a research framework for efficient feature selection and acquisition in machine learning scenarios where features have associated costs.

## 🚀 Main Novelties

The core contributions of this work are implemented in two key algorithms:

### 1. **Practical Search** 
📍 **Location**: `notebooks/template-serach/practical-search.py`
- **Function**: `tafalib.makers.templates.practical_serach_templates()` in `libs/internal/tafalib/makers/templates.py`
- **Description**: A greedy template generation method that creates feature selection templates by evaluating candidates and selecting the most cost-effective combinations

### 2. **Iterative Mutate Search**
📍 **Location**: `notebooks/template-serach/iterative-mutate-search.py` 
- **Function**: `tafalib.makers.templates.iterative_mutate_search_templates()` in `libs/internal/tafalib/makers/templates.py`
- **Description**: An advanced template generation approach that iteratively refines templates through mutation and selection, with optional feature importance sampling

## 📁 Repository Structure

```
tafa-demo/
├── datasets/                    # Dataset management and loading
│   ├── _files/                 # Raw dataset files (AACO, SEFA, etc.)
│   └── _python/                # Dataset loading utilities
├── experiments/                 # Experimental configurations
│   └── template/               # Hydra-based experiment templates
├── libs/internal/              # Core TAFA library
│   ├── mylib/                  # General ML utilities
│   └── tafalib/                # Main TAFA implementation
│       ├── functional.py       # Cost estimation functions
│       ├── utils.py           # Utility functions
│       └── makers/            # Template generation algorithms
│           ├── candidates.py   # Candidate generation
│           └── templates.py    # **Main novelty implementations**
├── models/                     # Model implementations
│   └── _python/               # Classifier implementations
├── notebooks/                  # Research notebooks and demos
│   ├── pretrain-classifier/   # Classifier pretraining
│   └── template-serach/       # **Main novelty demonstrations**
└── requirements.txt           # Python dependencies
```

## 🛠️ Setup and Installation

1. **Install dependencies**:
```bash
cd tafa-demo/
bash setup.sh
```

This script will:
- Install Python dependencies from `requirements.txt`
- Install local packages in development mode:
  - `datasets` package for data loading
  - `models` package for classifiers
  - `libs/internal` package for TAFA core functionality

## 🔬 Key Components

### Core Algorithm Implementation

The main algorithmic contributions are in `libs/internal/tafalib/makers/templates.py`:

- **`practical_serach_templates()`**: Implements the practical search algorithm for template generation
- **`iterative_mutate_search_templates()`**: Implements the iterative mutate search with feature importance sampling
- **`make_templates_from_candidates()`**: Core greedy selection logic for choosing optimal templates
- **`identify_init_fidx()`**: Identifies initial feature

### Cost Estimation Functions

Located in `libs/internal/tafalib/functional.py`:

- **`knn_cost_est()`**: K-nearest neighbor based cost estimation
- **`multi_output_nnet_cost_est()`**: Neural network based multi-output cost estimation
- **`single_output_nnet_cost_est()`**: Single-output neural network cost estimation

### Supported Datasets

The framework supports several datasets through `mydatasets.aaco`:
- `cube_20_0.3`: Synthetic cube dataset
- `big5_C_cls`: Big Five personality classification
- `gas`: Gas sensor dataset
- `mnist`: MNIST digit classification
- `volvo`: Volvo dataset

## 🧪 Running Experiments

### Practical Search Demo

`notebooks/template-serach/practical-search.py` is written with interactive-python, so you could run it in a jupyter notebook fashion with vscode.

This demonstrates:
- Template generation using practical search
- Feature acquisition with cost estimation
- Performance evaluation on validation data

### Iterative Mutate Search Demo

`notebooks/template-serach/iterative-mutate-search.py` is written with interactive-python, so you could run it in a jupyter notebook fashion with vscode.

This demonstrates:
- Iterative template refinement through mutation
- Feature importance sampling
- Multi-round optimization

### Key Parameters

Both algorithms support the following key parameters:
- `lmbda`: Feature cost penalty (e.g., 0.3 for cube, 0.075 for big5)
- `n_tmpls_targ`: Target number of templates
- `n_cands_targ`: Number of candidate templates to evaluate
- `init_fidx`: Initial feature index for bootstrapping
- `min_features`/`max_features`: Feature count constraints

## 📊 Evaluation Metrics

The framework evaluates performance using:
- **Classification metrics**: Accuracy, Precision, Recall, F1-Score, AUROC
- **Feature efficiency**: Average features observed vs. used
- **Cost effectiveness**: Trade-off between accuracy and feature acquisition cost

## 🔧 Extending the Framework

### Adding New Datasets
1. Add dataset files to `datasets/_files/`
2. Implement loader in `datasets/_python/mydatasets/`
3. Update dataset configurations

### Adding New Classifiers
1. Implement in `models/_python/mymodels/classifiers.py`
2. Ensure compatibility with `SubsetFeatureClassifier` protocol
3. Test with existing template generation methods

### Custom Cost Estimation
1. Implement new cost functions in `libs/internal/tafalib/functional.py`
2. Follow the signature pattern of existing estimators
3. Integrate with template evaluation pipeline

## 📈 Research Applications

This framework is designed for scenarios involving:
- **Medical diagnosis**: Where tests have associated costs
- **Sensor networks**: Where sensor activation incurs energy costs
- **Financial modeling**: Where data acquisition has monetary costs
- **Industrial monitoring**: Where measurements require resources

## 🔍 Citation

If you use this code in your research, please cite our paper:

```bibtex
[Citation details to be provided upon publication]
```

## 📝 Dependencies

Key dependencies include:
- PyTorch & TorchMetrics for deep learning
- Lightning for experiment management
- XGBoost for gradient boosting
- scikit-learn for ML utilities
- TensorDict for tensor management
- Hydra for configuration management

## 🤝 Contributing

For questions or contributions, please open an issue or submit a pull request.

---

**Note**: This repository contains the reference implementation for the TAFA research paper. The main algorithmic novelties—practical search and iterative mutate search—are specifically located in the template generation functions within the `tafalib.makers.templates` module.
