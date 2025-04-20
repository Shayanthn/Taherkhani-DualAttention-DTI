# Dual Attention CNN-GNN Model for Drug–Target Interaction Prediction

![DTI Banner](https://img.shields.io/badge/AI--Driven-Drug_Discovery-green.svg)

A cutting-edge deep learning framework developed by **Shayan Taherkhani** for predicting drug–target interactions (DTIs) using dual attention mechanisms on graph-based and sequence-based representations.

---

## 🌍 Overview
This repository contains:
- A fully implemented PyTorch pipeline to predict binding affinity between compounds and proteins.
- A custom Dual Attention architecture combining Graph Neural Networks (GNNs) for molecules and Convolutional Neural Networks (CNNs) for protein sequences.
- Integration of real-world benchmark dataset (Davis).
- Research article detailing model architecture and experimental results.

> **Paper**: [Dual_Attention_CNN-GNN_DTI_Model_Shayan_Taherkhani.pdf](Dual_Attention_CNN-GNN_DTI_Model_Shayan_Taherkhani.pdf)

---

## 📊 Key Features

- **Dual Attention Fusion**: Learns meaningful interactions between drug substructures and protein motifs.
- **GNN-Based Drug Encoding**: Uses `GATConv` layers on molecular graphs built from SMILES.
- **CNN-Based Protein Encoding**: Captures biochemical motifs using 1D convolutions.
- **Heatmap Visualization**: Built-in weight visualization for attention interpretability.
- **Benchmark Evaluation**: Tested on Davis dataset with CI, RMSE, and MSE metrics.

---

## 💡 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/shayanthn/Dual-Attention-DTI.git
cd Dual-Attention-DTI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Main Script
```bash
python dual_attention_dti_model.py
```

> Results will be printed in the terminal, and loss/metric curves + molecule visualization will be shown.

---

## 📊 Dataset: Davis
- Davis dataset provides experimental Kd binding affinities for kinase inhibitors.
- Format: `compound_smiles`, `target_sequence`, `affinity`
- Preprocessed as graphs (for drugs) and indexed amino acid sequences (for proteins).

---

## 🔄 Model Architecture

**Drug Encoder**: Graph Attention Network (GATConv)
**Protein Encoder**: Convolutional layers with varying kernel sizes
**Fusion**: Cross-attention (drug-to-protein and protein-to-drug)
**Output**: Regression for pKd prediction

---

## 🔹 Example Output

```
Epoch 50, Train Loss: 0.2167, Test Loss: 0.1884, RMSE: 0.4265, CI: 0.8764
True affinity: 8.50, Predicted: 8.42
```

## 🎓 Citation
```
@article{taherkhani2024dualattention,
  title={Dual Attention CNN-GNN Model for Drug--Target Interaction Prediction},
  author={Shayan Taherkhani},
  journal={Preprint},
  year={2024},
  url={https://github.com/shayanthn/Dual-Attention-DTI}
}
```

---

## 📄 License
This project is licensed under the **MIT License**. You are free to use, distribute, and modify the code for both academic and commercial purposes with attribution.

---

## 🚀 Future Work
- Transformer-based protein encoders
- Integration with 3D protein-ligand structures (e.g., AlphaFold)
- Enhanced cold-start prediction via meta-learning

---

## 📅 Maintainer
**Shayan Taherkhani**  
Email: [shayanthn78@gmail.com](mailto:shayanthn78@gmail.com)  
GitHub: [shayanthn](https://github.com/shayanthn)  
LinkedIn: [linkedin.com/in/shayantaherkhani](https://linkedin.com/in/shayantaherkhani)

---

If this repo helps your research, please consider giving it a star ⭐ and citing the preprint 🚀

