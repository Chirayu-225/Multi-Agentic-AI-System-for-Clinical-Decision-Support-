"""
src/features.py

Optional features, OFF by default. The core product is: a clinician reviews what changed in a
patient's labs, verified and in their language. These two are not part of that loop:

  ENABLE_FORECASTING  slope-based "will cross threshold by <date>" projections. A linear fit on a
                      handful of visits is illustrative at best, and a date shown to a clinician
                      reads as a prediction. Kept for research, hidden in the product.
  ENABLE_PUBMED       literature links for HIGH/CRITICAL findings. Rarely read in a review
                      workflow, and it costs an external call per report.

Read at call time (not import time) so a restart is not needed to change them in tests.
"""
import os


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in ("1", "true", "yes", "on")


def forecasting_enabled() -> bool:
    return _flag("ENABLE_FORECASTING")


def pubmed_enabled() -> bool:
    return _flag("ENABLE_PUBMED")
