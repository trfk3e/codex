"""Utility to track Gemini API usage across projects and API keys.

Each generated text requires two API calls: one to get a topic and one
for the body.  Different Gemini versions provide different daily quota
of ready texts per API key:

* ``gemini-2.5-pro`` – 25 texts per key
* ``gemini-2.5-flash`` or ``gemini-2.5-flash-lite`` – 250 texts per key

The :class:`QuotaManager` helps keep a global count of texts requested
by simultaneously running projects and prevents exceeding the total
available quota.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


_MODEL_LIMITS = {
    "gemini-2.5-pro": 25,  # texts per API key
    "gemini-2.5-flash": 250,
    "gemini-2.5-flash-lite": 250,
}


@dataclass
class QuotaManager:
    """Tracks quota usage across projects.

    Parameters
    ----------
    model : str
        Gemini model name. Only supported values are keys of
        ``_MODEL_LIMITS``.
    api_keys : int
        Number of available API keys.
    """

    model: str
    api_keys: int
    _project_usage: Dict[str, int] = field(default_factory=dict, init=False)
    _used_texts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.model not in _MODEL_LIMITS:
            raise ValueError(f"Unsupported model: {self.model!r}")
        if self.api_keys < 1:
            raise ValueError("api_keys must be positive")

    @property
    def capacity(self) -> int:
        """Total number of texts that can be generated."""
        return _MODEL_LIMITS[self.model] * self.api_keys

    @property
    def used(self) -> int:
        """Number of texts currently reserved by running projects."""
        return self._used_texts

    def start_project(self, name: str, texts_needed: int) -> None:
        """Reserve quota for a new project.

        Raises
        ------
        RuntimeError
            If starting the project would exceed the total quota.
        """
        if texts_needed < 0:
            raise ValueError("texts_needed must be non-negative")
        if self._used_texts + texts_needed > self.capacity:
            raise RuntimeError(
                "Quota exceeded: requested "
                f"{self._used_texts + texts_needed} texts, but only "
                f"{self.capacity} are available"
            )
        self._project_usage[name] = texts_needed
        self._used_texts += texts_needed

    def finish_project(self, name: str) -> None:
        """Release quota reserved by a project."""
        texts = self._project_usage.pop(name, 0)
        self._used_texts = max(0, self._used_texts - texts)

    def remaining(self) -> int:
        """Return number of texts that can still be generated."""
        return self.capacity - self._used_texts


if __name__ == "__main__":
    # Example usage reflecting the question description.
    qm = QuotaManager("gemini-2.5-flash", api_keys=1)
    try:
        qm.start_project("project", texts_needed=950)
    except RuntimeError as exc:
        print(f"Cannot start project: {exc}")

    qm = QuotaManager("gemini-2.5-flash", api_keys=100)
    qm.start_project("p1", texts_needed=12000)
    qm.start_project("p2", texts_needed=12000)
    try:
        qm.start_project("p3", texts_needed=2000)
    except RuntimeError as exc:
        print(f"Third project blocked: {exc}")
