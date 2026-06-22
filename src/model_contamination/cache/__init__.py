"""logprobs / generate 结果缓存，避免重跑差分时重算。"""

from model_contamination.cache.store import CacheStore

__all__ = ["CacheStore"]
