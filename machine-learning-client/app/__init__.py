"""ML Client package

Keep package-level imports minimal so importing `app` does not execute
long-running or environment-specific code (like `worker` which expects
container paths and services). Import `worker` only when needed.
"""

# pylint: disable=import-error
from .features import extract_features, cosine_sim

__all__ = ["extract_features", "cosine_sim"]
