

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

```bash
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
```

## 🚀 Getting Started

## 🐍 Environment Requirements

* **Python version:** 3.10.x (required)

### Step 1: Clone the Repository

```bash
git clone https://github.com/emirmasood/HeatNet.git
cd HeatNet
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Data and Models

Due to GitHub's file size restrictions, download large files separately:

* **Dataset:** [Google Drive Data Folder](https://drive.google.com/drive/folders/1bMuIT9NpPXCQPV6SGFvr6aIEn42B3BZ-?usp=sharing)
* **ResNet Checkpoints:** [ResNet Checkpoints](https://drive.google.com/drive/folders/14pTckwpHFnaL27vCwQ3DRbv9XOCgZZOM?usp=drive_link)
* **YOLOv10m pretrained weights:** [YOLOv10m Checkpoint](https://drive.google.com/file/d/1mRdriU3u85oxcL0CPeIhJBxX795iENse/view?usp=drive_link)

Place downloaded files into their respective folders as indicated in the folder structure above.

### Step 4: 
- Run baseline
- Run extension1
- Run extension2


## 👥 Authors
This project was created by:

Ana Parovic (ana.parovic@studenti.polito.it)

Antonio Potenza (antonio.potenza@studenti.polito.it)

Era Alcani (era.alcani@studenti.polito.it)

Ismail Aljosevic (ismail.aljosevic@studenti.polito.it)

Nicoletta Toma (nicoletta.toma@studenti.polito.it)

```





