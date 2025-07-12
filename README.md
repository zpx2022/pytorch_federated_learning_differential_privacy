[English](README.md) | [简体中文](README.zh-CN.md)
# Combination of Federated Learning Baseline Algorithm and Laplace Mechanism-based Differential Privacy Technology

**Project Description：** This project is a fork of [rruisong/pytorch_federated_learning](https://github.com/rruisong/pytorch_federated_learning). I would like to express my gratitude to the original author for his basic work.

## ✨ Core Features

Compared with the parent project, this project implements privacy protection based on the Laplace mechanism and makes a series of architectural optimizations:
- **Support for Local Differential Privacy (LDP)**: The LDP mechanism is integrated, and gradient clipping and Laplacian noise can be added by configuration to study federated learning under privacy protection.

- **Efficient parallel simulation**: The client's **parallel training** is implemented using `concurrent.futures`, which can make full use of multi-core CPU resources and significantly shorten the simulation time.

- **Early Termination**: Supports the Early Stopping mechanism, which automatically stops training when the model performance no longer improves within the set patience value, saving computing resources.

- **Breakpoint Resume**: It can automatically save and load training checkpoints (`checkpoint`), making it easy to resume long experiments from interruptions.

## ⚙️ Project structure

```text
├── fed_baselines/ # Core algorithm implementation
│ ├── client_base.py # Client base class (FedAvg)
│ ├── server_base.py # Server base class (FedAvg)
│ └── ... # Other algorithm implementations
├── preprocessing/ # Data preprocessing
│ └── baselines_dataloader.py # Data loading and Non-IID partitioning
├── postprocessing/ # Result postprocessing
│ └── recorder.py # Result recording and drawing
├── utils/ # Auxiliary tools
│ ├── models.py # Model definition
│ └── fed_utils.py # Auxiliary functions
├── fl_main.py # Main training program
├── eval_main.py # Result evaluation program
├── test_config.yaml # Experiment configuration file
└── README.md
```

## 🚀 Quick Start

### 1. Environment preparation

It is recommended to use a virtual environment (such as `conda` or `venv`) to manage project dependencies.

```bash
# Clone the repository
git clone https://github.com/zpx2022/pytorch_federated_learning_differential_privacy.git
cd pytorch_federated_learning_differential_privacy

# (Optional, recommended) Create and activate the conda virtual environment
conda create -n fldp python=3.8
conda activate fldp

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure the experiment
Open the test_config.yaml file and modify the experimental parameters according to your needs.

### 3. Run training
Execute the main program fl_main.py to start training. All results and checkpoints will be saved in the results/ and checkpoints/ directories by default.
```bash
python fl_main.py --config test_config.yaml
```

### 4. Evaluate and visualize results
Draw all .json result files in the results/ directory into a graph
```bash
python eval_main.py --sys-res_root results
```
