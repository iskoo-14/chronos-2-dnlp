# config.py

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train.csv")
STORE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "store.csv")

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

HORIZON = 30
CONTEXT_LENGTHS = [128, 256, 512]
BEST_CONTEXT = 512

PAST_ONLY_COVS = ["Customers"]
FUTURE_KNOWN_COVS = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]

KEEP_CLOSED_DAYS = True
ENFORCE_DAILY_FREQUENCY = True

SAVE_FUTURE_DEBUG = True

# doesn't do preprocessing again if it's already done
SKIP_EXISTING_PROCESSED = True     

# doesn't do forecast again if it's already done
SKIP_EXISTING_FORECASTS = True

# doesn't do debug again if it's already done
SKIP_EXISTING_GT_DEBUG = True
