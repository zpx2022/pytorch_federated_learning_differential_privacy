### **About this Fork**

**This project is a fork of [rruisong/pytorch_federated_learning](https://github.com/rruisong/pytorch_federated_learning).** The main contribution of this fork is the implementation of **client-side Differential Privacy (DP)** to study the privacy-utility trade-off in Federated Learning.

**Key Modifications:**
* Integrated a **Laplace noise mechanism** into the client-side training logic to perturb model weights before they are uploaded to the server.
* Made the Differential Privacy feature flexible via the `test_config.yaml` file, allowing users to easily enable/disable it and adjust the noise intensity.
* Conducted systematic, comparative experiments to quantify the impact of different privacy levels on model performance.

---

### **Key Experimental Results**

This project evaluates the effectiveness of Differential Privacy in a challenging **pathological Non-IID setting (2 classes per client)**. The plot below shows the performance of the FedAvg algorithm on the MNIST dataset under different levels of Laplace noise intensity (`b`).

![Federated Learning LDP Comparison](<figures/FedAvg_LeNet_MNist_NIID_LDP_Comparison_Annotated.png>)

**Experimental Conclusions:**

| Noise Intensity (b) | Max Accuracy | Accuracy Drop vs. Baseline | Round of Max Accuracy |
| :--- | :--- | :--- | :--- |
| 0.00 (No Noise) | **98.75%** | - | 1962 |
| 0.01 | **98.33%** | 0.42% | 1706 |
| 0.03 | **97.42%** | 1.33% | 1939 |
| 0.05 | **95.56%** | 3.19% | 1575 |

The data clearly illustrates the non-linear trade-off between privacy and utility. Notably, at a noise intensity of 0.01, the system can achieve effective privacy protection at the minimal cost of only a 0.42% drop in accuracy, identifying a practical balance point for real-world deployment.





