import subprocess
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def run_if_missing(output_path, script_path):
    """Run script only if expected output does not already exist."""
    if os.path.exists(output_path):
        print(f"[SKIPPED] {script_path} (output already exists: {output_path})")
        return
    print(f"\n=== Running {script_path} ===")
    subprocess.run(["python", script_path], check=True)
    print(f"=== Finished {script_path} ===\n")

if __name__ == "__main__":
    
    preprocess = os.path.join(BASE, "docs/data_preparation/prepare_rossmann.py")
    univariate = os.path.join(BASE, "docs/forecasting/run_univariate.py")
    covariates = os.path.join(BASE, "docs/forecasting/run_covariates.py")
    robustness = os.path.join(BASE, "docs/experiments/run_robustness_tests.py")

    print("=== DNLP PROJECT PIPELINE STARTED ===")

    # 1. Preprocessing
    processed_file = os.path.join(BASE, "docs/data_preparation/processed_rossmann.csv")
    run_if_missing(processed_file, preprocess)

    # 2. Univariate
    univariate_output = os.path.join(BASE, "docs/forecasting/univariate_results.csv")
    run_if_missing(univariate_output, univariate)

    # 3. Covariates
    cov_output = os.path.join(BASE, "docs/forecasting/covariate_results/forecast.csv")
    run_if_missing(cov_output, covariates)

    # 4. Robustness
    robustness_output = os.path.join(BASE, "docs/experiments/noise_test/forecast.csv")
    run_if_missing(robustness_output, robustness)

    print("=== PIPELINE COMPLETED ===")
