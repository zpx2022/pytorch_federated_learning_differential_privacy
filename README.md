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
├── config/                   # Storage of configuration files
│   └── test_config.yaml      # Main experimental configuration files
├── fed_baselines/            # Core algorithm implementation
│   ├── client_base.py        # Client base class (implementation of FedAvg)
│   ├── client_fedprox.py     # FedProx algorithm client
│   ├── client_scaffold.py    # SCAFFOLD algorithm client
│   ├── client_fednova.py     # FedNova client
│   ├── server_base.py        # Server base class (implementation of FedAvg)
│   ├── server_scaffold.py    # SCAFFOLD algorithm server
│   └── server_fednova.py     # FedNova algorithm server
├── figures/                  # Storage of generated charts
├── postprocessing/           # Result post-processing
│   ├── eval_main.py          # Main program for result evaluation and visualization
│   └── recorder.py           # Result recording and drawing tool class
├── preprocessing/            # Data preprocessing
│   └── baselines_dataloader.py # Data loading and Non-IID partitioning
├── utils/                    # Auxiliary tools
│   ├── models.py             # Neural network model definition
│   └── fed_utils.py          # Federated learning auxiliary function
├── fl_main.py                # Federated learning main training program
└── requirements.txt          # Python dependency package list
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
