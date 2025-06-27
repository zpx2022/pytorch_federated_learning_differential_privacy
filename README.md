# PyTorch Implementation of Federated Learning with Differential Privacy

[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-278ea5)](...) ![Stars](https://img.shields.io/github/stars/rruisong/pytorch_federated_learning?color=yellow&label=Stars) ![Forks](https://img.shields.io/github/forks/rruisong/pytorch_federated_learning?color=green&label=Forks)

---

### **About this Fork**

**This project is a fork of [rruisong/pytorch_federated_learning](https://github.com/rruisong/pytorch_federated_learning).** The main contribution of this fork is the implementation of **client-side Differential Privacy (DP)** to study the privacy-utility trade-off in Federated Learning.

**Key Modifications:**
* Integrated a **Laplace noise mechanism** into the client-side training logic to perturb model weights before they are uploaded to the server.
* Made the Differential Privacy feature flexible via the `test_config.yaml` file, allowing users to easily enable/disable it and adjust the noise intensity.
* Conducted systematic, comparative experiments to quantify the impact of different privacy levels on model performance.

---

### **Key Experimental Results**

This project evaluates the effectiveness of Differential Privacy in a challenging **pathological Non-IID setting (2 classes per client)**. The plot below shows the performance of the FedAvg algorithm on the MNIST dataset under different levels of Laplace noise intensity (`b`).

![Federated Learning LDP Comparison](figures/FedAvg_MNIST_NonIID_LDP_Comparison_Annotated.png)

**Experimental Conclusions:**

| Noise Intensity (b) | Max Accuracy | Accuracy Drop vs. Baseline | Round of Max Accuracy |
| :--- | :--- | :--- | :--- |
| 0.00 (No Noise) | **98.75%** | - | 1962 |
| 0.01 | **98.33%** | 0.42% | 1706 |
| 0.03 | **97.42%** | 1.33% | 1939 |
| 0.05 | **95.56%** | 3.19% | 1575 |

The data clearly illustrates the non-linear trade-off between privacy and utility. Notably, at a noise intensity of 0.01, the system can achieve effective privacy protection at the minimal cost of only a 0.42% drop in accuracy, identifying a practical balance point for real-world deployment.

---
*The following is the original README content, adapted for this project.*

## PyTorch Implementation of Federated Learning Baselines

PyTorch-Federated-Learning provides various federated learning baselines implemented using the PyTorch framework. The codebase follows a client-server architecture and is highly intuitive and accessible.

[English](README.md) | [简体中文](README.zh-CN.md)

* **Current Baseline implementations**: Pytorch implementations of the federated learning baselines. The currently supported baselines are FedAvg, FedNova, FedProx and SCAFFOLD.
* **Dataset preprocessing**: Downloading the benchmark datasets automatically and dividing them into a number of clients w.r.t. federated settings.
* **Postprocessing**: Visualization of the training results for evaluation.

## Installation

### Dependencies

 - Python (3.8)
 - PyTorch (1.8.1)
 - OpenCV (4.5)
 - numpy (1.21.5)
 - matplotlib

### Install requirements

Run: `pip install -r requirements.txt` to install the required packages.

## Execute the Federated Learning Baselines

Hyperparameters are defined in a yaml file, e.g. `./config/test_config.yaml`. **This fork adds `use_ldp` and `ldp_noise_scale` options to control the differential privacy mechanism.**

**Example `test_config.yaml`:**
```yaml
client:
  # ... other params
  use_ldp: True               # Set to True to enable LDP, False to disable.
  ldp_noise_scale: 0.01       # Controls the intensity of the Laplace noise.
```

Run the experiment with this configuration:
```
python fl_main.py --config "./config/test_config.yaml"
```

## Evaluation Procedures

Please run `python postprocessing/eval_main.py -rr 'results'` or the custom `python postprocessing/plot_all_results.py` script to plot the testing accuracy and training loss.

## Citation

*Please cite the original authors' work if you use their baseline implementations.*
```
