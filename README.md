

<p align="center">
  <img src="https://i.ytimg.com/vi/x5sOms9SCHU/maxresdefault.jpg" width="500"/>
</p>


# Zero-Shot Time Series Forecasting on Rossmann Sales Data using Chronos-2 Transformers

## 📌 Overview

The project investigates how well a large pre-trained time-series foundation model generalizes to retail sales forecasting without task-specific fine-tuning.

Beyond baseline zero-shot inference, the study explores systematic improvements through:

* Feature engineering
* Clustering-based store segmentation
* Identity-aware residual modeling

The goal is to analyze both the strengths and limitations of foundation models in structured retail forecasting tasks and to design hybrid strategies that enhance predictive performance.

### 🔎 Main Contributions

1. **Feature-Enhanced Zero-Shot Forecasting** – We improve baseline zero-shot inference by introducing cyclical seasonality encodings, administrative markers, and momentum/volatility signals to strengthen in-context learning.
2. **Store Regime Discovery via Clustering** – We identify latent subgroups of stores with similar operational dynamics using unsupervised clustering analysis.
3. **Regime-Aware Model Probing** – We propose a novel probing strategy that enables Chronos-2 to leverage information from stores with similar working regimes without fine-tuning model parameters.

---

## 📂 Project Structure
chronos-2-dnlp/
│
├── baseline/            # Baseline model implementation
├── data/                # Dataset and preprocessing scripts
├── evaluation/          # Evaluation metrics and testing scripts
├── extension1/          # First project extension
├── extension2/          # Second project extension
├── models/              # Model architectures and saved weights
├── notebooks/           # Jupyter notebooks for experiments
├── outputs/             # Predictions and generated outputs
├── reports/             # Reports and documentation
├── visualization/       # Data and result visualizations
│
├── config.py            # Project configuration file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

## 🐍 Environment Requirements

* **Python version:** 3.10.x (required)

### Core Dependencies

See `requirements.txt` for the full list of dependencies.
---

---

