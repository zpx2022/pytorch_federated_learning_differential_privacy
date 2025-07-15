[English](README.md) | [简体中文](README.zh-CN.md)
# Combination of Federated Learning and Laplace Mechanism-based Differential Privacy Technology

**Project Description：** This project is a fork of [rruisong/pytorch_federated_learning](https://github.com/rruisong/pytorch_federated_learning). I would like to express my gratitude to the original author for his basic work.

## ✨ Core Features

Compared with the parent project, this project implements privacy protection based on the Laplace mechanism and makes a series of architectural optimizations:
- **Support for Local Differential Privacy (LDP)**: The LDP mechanism is integrated, and gradient clipping and Laplacian noise can be added by configuration to study federated learning under privacy protection.

- **Efficient parallel simulation**: The client's **parallel training** is implemented using `concurrent.futures`, which can make full use of multi-core CPU resources and significantly shorten the simulation time.

- **Early Termination**: Supports the Early Stopping mechanism, which automatically stops training when the model performance no longer improves within the set patience value, saving computing resources.

- **Breakpoint Resume**: It can automatically save and load training checkpoints (`checkpoint`), making it easy to resume long experiments from interruptions.
- **Elegant architecture design**: The client architecture was reconstructed using the "template method" design pattern, eliminating the redundancy of repeatedly writing training loops in each algorithm, significantly improving the maintainability and scalability of the code.

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

## 📈 Experimental results and analysis

![Comparison of accuracy and loss curves for different noise intensities](figures/FedAvg_LeNet_MNIST_NIID_LDP_Comparison_Annotated.png)

I conducted experiments on the highly non-IID MNIST dataset (each client only has 2 categories of data) to evaluate the impact of different intensities of Laplace noise (`laplace_noise_scale` from `0.0` to `0.1`) on the performance of the FedAvg algorithm.
...

The basis for selecting Laplace noise intensity: MNist is a handwritten dataset with 10 categories (0-9), which is low-sensitivity data. We can adopt a utility-first strategy, that is, the `max acc` loss is required to be less than `0.01`, and the privacy budget is `>=10`. According to the formula of privacy budget = `sensitivity`/`noise intensity` (sensitivity is similar to the value of the hyperparameter `grad_clip_norm`, which is `1.0`), the noise intensity should be `<=0.1`. To narrow the range, the privacy budget is set to `<=20`. At this time, the noise intensity range is `[0.05,0.1]`, and `0.05`, `0.06`, `0.07`, `0.08`, `0.09`, `0.1` are selected.

### 1\. Comparison of test accuracy

The first sub-figure shows how the test accuracy of the global model changes with the number of communication rounds under different noise intensities. From the accuracy curve, we can clearly observe the following points:

* **Privacy-utility trade-off**: The overall trend shows that as the noise intensity increases, the final convergence performance (maximum accuracy) of the model will decrease. This intuitively demonstrates the classic "privacy-utility trade-off" in differential privacy - in order to enhance privacy protection, part of the model's performance needs to be sacrificed. Based on the utility-first strategy, the `max acc` loss of experiments with noise intensities of `0.05`, `0.06`, and `0.07` is less than `0.01`, which can all be used as effective local differential privacy parameter settings.

* **Convergence speed**: In general, the larger the noise intensity of the experiment, the smaller the rounds corresponding to the model reaching the final convergence performance (maximum accuracy), that is, **the faster it reaches the performance bottleneck**.

### 2\. Training loss comparison

The second sub-figure shows the impact of different noise intensities on the average training loss of the global model. The trend of the loss curve is consistent with the accuracy curve:

* **Fitting difficulty**: The higher the noise intensity, the higher the training loss, indicating that the injected noise increases the difficulty of the model fitting local data and affects the learning process.

* **Stability**: The loss curve of the low noise (such as `0.0`, `0.05`) experiment drops more smoothly, while the curve of the high noise experiment shows more violent fluctuations.
