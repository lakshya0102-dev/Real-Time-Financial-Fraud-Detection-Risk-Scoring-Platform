"""Train the fraud detection model. Wrapper for training/run_experiments.py."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.run_experiments import main

if __name__ == "__main__":
    main()
