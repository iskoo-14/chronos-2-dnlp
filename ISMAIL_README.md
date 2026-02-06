## Shop-Level Dataset for Clustering (Extension 2)

At this stage of the project, a **shop-level dataset** has been introduced to enable
**clustering of stores** based on their historical sales behavior.

---

### Implementation Details

- A new function, **`aggregate_shop`**, has been added to the
  `data/make_dataset.py` module.  
  This function aggregates time-series data of individual shops into a
  **fixed-length feature vector**, capturing key statistical, temporal,
  seasonal, and promotional characteristics.

- The script **`run_ext2.py`** is responsible for dataset creation. It:
  - iterates over preprocessed CSV files located in `data/extension1`,
  - applies the `aggregate_shop` function to each shop,
  - constructs a consolidated shop-level dataset that will be used for
    **clustering analysis**.

---

### Dataset Description

The resulting dataset contains **one row per shop**, where each row represents
an aggregated summary of the shop’s historical behavior.

The dataset includes the following groups of features:

- **Sales level and variability**
  (e.g. mean, standard deviation, coefficient of variation),
- **Short-term dynamics** and spike behavior,
- **Temporal dependencies** via lag-based correlations,
- **Seasonality strength** (weekly, monthly, yearly),
- **Promotion and holiday effects**,
- **Store operation patterns** (open/closed behavior).

---

### Output Location

The generated dataset is stored in:

```text
data/extension2/shop_features_for_clustering.csv
