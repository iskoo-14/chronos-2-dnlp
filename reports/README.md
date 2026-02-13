# Reports

This folder contains all evaluation results and summary reports generated for the Baseline, Extension 1, and Extension 2 pipelines.

---

## 📂 Folder Structure

```bash

reports/
│
├── baseline/ # Full evaluation results for Baseline
├── baseline_test_only/ # Baseline results evaluated strictly on the test set
│
├── extension1/ # Full evaluation results for Extension 1
├── extension1_test_only/ # Extension 1 results evaluated strictly on the test set
├── extension1_clustering_analysis/ # Cluster-level evaluation and analysis
│
├── extension2_test_only/ # Extension 2 results evaluated strictly on the test set
│
├── store_validity.csv
├── store_validity_summary.csv
├── valid_store_ids.txt
├── wql_summary_all_together.csv
│
└── README.md
```
---

## 📌 Important Note on Evaluation

The folders:

- `baseline_test_only/`
- `extension1_test_only/`
- `extension2_test_only/`

contain results computed **exclusively on the test set**.

These evaluations were generated to ensure a **fair and comparable assessment** across all three approaches (relevant for Extension 2 Performance).

---

## 📊 Additional Files

- `wql_summary_all_together.csv` – Aggregated comparison of WQL metrics across all approaches.
- `store_validity.csv` / `store_validity_summary.csv` – Store-level validity checks.
- `valid_store_ids.txt` – List of stores included in the final evaluation.

All reports are reproducible from their corresponding pipeline scripts.
