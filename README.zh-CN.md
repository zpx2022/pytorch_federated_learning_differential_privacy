[English](README.md) | [简体中文](README.zh-CN.md)
# 联邦学习和基于拉普拉斯机制的差分隐私技术的结合

**项目说明：** 本项目是 [rruisong/pytorch_federated_learning](https://github.com/rruisong/pytorch_federated_learning) 的一个分支，本人对原作者的基础性工作表示感谢。

## ✨ 核心特性

相较于父项目，本项目实现了基于拉普拉斯机制的隐私保护，做出了一系列的架构优化：
- **支持本地差分隐私 (LDP)**: 集成了 LDP 机制，可通过配置开启梯度裁剪 (Gradient Clipping) 和拉普拉斯噪声 (Laplacian Noise) 添加，用于研究隐私保护下的联邦学习。

- **高效的并行模拟**: 利用 `concurrent.futures` 实现了客户端的**并行训练**，可充分利用多核 CPU 资源，显著缩短模拟所需时间。

- **提前终止**: 支持提前终止 (Early Stopping) 机制，当模型性能在设定的耐心值内不再提升时自动停止训练，节约计算资源。

- **断点续传**: 能够自动保存和加载训练检查点 (`checkpoint`)，方便从中断处恢复长时间的实验。
  
- **优雅的架构设计**: 采用“模板方法”设计模式重构了客户端架构，消除了在每个算法中重复编写训练循环的冗余问题，显著提升了代码的可维护性与扩展性。

## ⚙️ 项目结构

```text
├── config/                    # 存放配置文件
│   └── test_config.yaml       # 主要实验配置文件
├── fed_baselines/             # 核心算法实现
│   ├── client_base.py         # 客户端基类 (实现 FedAvg)
│   ├── client_fedprox.py      # FedProx 算法客户端
│   ├── client_scaffold.py     # SCAFFOLD 算法客户端
│   ├── client_fednova.py      # FedNova 算法客户端
│   ├── server_base.py         # 服务器基类 (实现 FedAvg)
│   ├── server_scaffold.py     # SCAFFOLD 算法服务器
│   └── server_fednova.py      # FedNova 算法服务器
├── figures/                   # 存放生成的图表
├── postprocessing/            # 结果后处理
│   ├── eval_main.py           # 结果评估与可视化主程序
│   └── recorder.py            # 结果记录与绘图工具类
├── preprocessing/             # 数据预处理
│   └── baselines_dataloader.py# 数据加载与 Non-IID 划分
├── utils/                     # 辅助工具
│   ├── models.py              # 神经网络模型定义
│   └── fed_utils.py           # 联邦学习辅助函数
├── fl_main.py                 # 联邦学习主训练程序
└── requirements.txt           # Python 依赖包列表
```

## 🚀 快速开始

### 1. 环境准备

建议使用虚拟环境（如 `conda` 或 `venv`）来管理项目依赖。

```bash
# 克隆仓库
git clone https://github.com/zpx2022/pytorch_federated_learning_differential_privacy.git
cd pytorch_federated_learning_differential_privacy

# (可选，推荐) 创建并激活 conda 虚拟环境
conda create -n fldp python=3.8
conda activate fldp

# 安装依赖
pip install -r requirements.txt
```

### 2.配置实验
打开 test_config.yaml 文件，根据您的需求修改实验参数。

### 3.运行训练
执行主程序 fl_main.py 开始训练。所有结果和检查点将默认保存在 results/ 和 checkpoints/ 目录下。
```bash
python fl_main.py --config test_config.yaml
```

### 4.评估与可视化结果
训练结束后，使用 eval_main.py 来绘制性能曲线图。
```bash
# 将 results/ 目录下的所有 .json 结果文件绘制成图
python eval_main.py --sys-res_root results
```
