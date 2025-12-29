Changes were made in make_dataset.py 
Single store analysis was performed in single_analysis.py

Note: The current code is not well-structured and includes ad-hoc modifications introduced during debugging and experimentation. I apologize in advance.

1. Steps Performed

- Filtered the original dataset to include only one store

- Sorted observations in ascending chronological order

- Kept days when the store is closed

- Covariate selection (Used only the set of covariates reported in the paper)

- Adjustment of future covariates

- Ensured compatibility with the Chronos model requirements

- Implemented the prediction pipeline from the official Chronos GitHub repository

