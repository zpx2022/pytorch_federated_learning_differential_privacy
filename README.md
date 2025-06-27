# PyTorch Implementation of Federated Learning with Differential Privacy

[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-278ea5)](...) ![Stars](https://img.shields.io/github/stars/rruisong/pytorch_federated_learning?color=yellow&label=Stars) ![Forks](https://img.shields.io/github/forks/rruisong/pytorch_federated_learning?color=green&label=Forks)

---

### **关于此 Fork (About this Fork)**

**本项目是 [rruisong/pytorch_federated_learning](https://github.com/rruisong/pytorch_federated_learning) 的一个二次开发版本。** 我在此基础上，主要增加了**客户端差分隐私（Client-Side Differential Privacy）**的实现，以用于研究联邦学习中的隐私-效用权衡问题。

**主要修改点 (Key Modifications):**
* 在客户端训练逻辑中，集成了**拉普拉斯噪声机制**，可在模型上传前对权重进行扰动。
* 通过配置文件 (`test_config.yaml`) 可灵活开启/关闭差分隐私，并调整噪声强度。
* 开展了系统的对比实验，以量化不同隐私保护等级对模型性能的影响。

---

### **核心实验成果展示 (Key Experimental Results)**

本项目在病态非独立同分布（Non-IID, 每客户端2类）的挑战性环境下，对差分隐私的效用进行了评估。下图展示了在不同拉普拉斯噪声强度 (`b`)下，FedAvg算法在MNIST数据集上的性能表现：

![Federated Learning LDP Comparison](<figures/FedAvg_LeNet_MNist_NIID_LDP_Comparison_Annotated.png>)

**实验结论 (Conclusions):**

| 噪声强度 (b) | 最高准确率 (Max Acc) | 相比无噪声的准确率下降 | 首次达到最高准确率的轮次 |
| :--- | :--- | :--- | :--- |
| 0.00 (无噪声) | **98.75%** | - | 1962 |
| 0.01 | **98.33%** | 0.42% | 1706 |
| 0.03 | **97.42%** | 1.33% | 1939 |
| 0.05 | **95.56%** | 3.19% | 1575 |

从数据中可以清晰地看到隐私与效用之间的非线性权衡关系。特别地，在噪声强度为0.01时，系统可在仅牺牲0.42%准确率的微小代价下，获得有效的隐私保护，找到了一个实际可行的平衡点。


