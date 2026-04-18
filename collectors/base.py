"""Base collector class and data types."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawCollectedItem:
    """A raw piece of text collected from an external source."""
    text: str
    source: str  # e.g., "github", "reddit", "arxiv"
    url: str = ""
    title: str = ""
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class BaseCollector(ABC):
    """Abstract base for all external data collectors."""

    source_name: str = "unknown"

    @abstractmethod
    def collect(self, query: str = "LLM jailbreak prompt",
                max_items: int = 20) -> list[RawCollectedItem]:
        """Collect raw items from the source. Returns list of RawCollectedItem."""
        ...

    def __repr__(self):
        return f"<{self.__class__.__name__} source={self.source_name}>"
