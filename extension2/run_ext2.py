import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_ext1 import INCLUDE_CHG, INCLUDE_EMA, INCLUDE_ROLLING

print(INCLUDE_CHG)
