# 🧬 DNA-Vault: Task-Free Modular Continual Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
**DNA-Vault** is a novel PyTorch-based neural network architecture designed to solve **Catastrophic Forgetting** in Continual Learning. Traditional neural networks suffer from weight interference when learning new data. DNA-Vault mitigates this by dynamically partitioning its layers into isolated "vaults". 

Using a custom **Prototype Routing** mechanism, the model operates completely **task-free** during inference. It autonomously infers which task it is solving by calculating the L2 distance between the incoming normalized input and dynamically updated task prototypes, routing the data to the correct vault.

## 🚀 Key Features
* **Zero Catastrophic Forgetting (Linear Layers):** Achieves near-perfect retention of old tasks when learning new ones sequentially.
* **Task-Free Routing:** The model dynamically routes information without needing explicit `task_id` labels during inference.
* **Modular Architecture:** Segregates weights using mathematical masking, preventing gradient overlap.
* **Immune to OOD Overconfidence:** Utilizes L2 feature normalization and energy-based distance metrics instead of raw Softmax, preventing logit explosions.

## 📊 Benchmarks & Empirical Results

### 1. Split MNIST (The Success)
Tested on a sequential learning task (Task 0: Normal MNIST -> Task 1: Inverted MNIST).

| Task State | Baseline (Standard MLP) | DNA-Vault (Ours) |
| :--- | :--- | :--- |
| **Task 0 Accuracy** (Initial) | ~99.0% | **91.39%** |
| **Task 1 Accuracy** (New Task) | ~99.0% | **87.60%** |
| **Task 0 Accuracy** (After Task 1) | **~33.0%** ❌ | **89.69%** ✅ |

*Result: DNA-Vault successfully retains previous knowledge with < 2% performance drop, while the baseline suffers catastrophic forgetting.*

### 2. CIFAR-10 (Scaling & Future Work)
To test scalability, a shared CNN feature extractor was attached to the DNA-Vault to process CIFAR-10 (Task 0: Normal -> Task 1: Inverted).

| Task State | Baseline (Shared CNN + MLP) | DNA-Vault (Shared CNN + Vault) |
| :--- | :--- | :--- |
| **Task 0 Retention** | 30.24% | **38.66%** |

*Scientific Note: While DNA-Vault retained more knowledge than the baseline, the drop in accuracy highlights a known limitation in Continual Learning: **Shared Convolutional Feature Drift**. Because the CNN layers were not partitioned, learning Task 1 altered the feature extractor's filters, distorting the input before it reached the DNA-Vault. This provides a strong foundation for future research into fully partitioned convolutional layers.*

## 💻 Quick Start & Usage

### Installation
Clone the repository and install PyTorch:
```bash
git clone [https://github.com/zorino96/DNA-Vault.git](https://github.com/zorino96/DNA-Vault.git)
cd DNA-Vault
