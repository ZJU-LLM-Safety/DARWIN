"""Model package.

Avoid importing heavyweight backends at package import time.
"""

__all__ = [
    "api_model",
    "llm_manager",
    "local_model",
]
