<img src="https://i.ytimg.com/vi/x5sOms9SCHU/maxresdefault.jpg" width="400">



# Zero-Shot Time Series Forecasting on Rossmann Sales Data using Chronos-2 Transformers

## 📌 Overview

This repository presents the **zero-shot forecasting performance** of the Chronos-2 Transformer model on the Rossmann Store Sales dataset. The project investigates how well a large pre-trained time-series foundation model generalizes to retail sales forecasting without task-specific fine-tuning.

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

## 🐍 Environment Requirements

* **Python version:** 3.10.x (required)

### Core Dependencies

See `requirements.txt` for the full list of dependencies.
---

## 📊 Dataset

This project uses the **Rossmann Store Sales dataset**, which contains historical daily sales data for Rossmann drug stores.
---

