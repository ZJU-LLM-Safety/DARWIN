"""Institution report collector — scrape AI safety org blogs."""
import requests
from collectors.base import BaseCollector, RawCollectedItem


class InstitutionCollector(BaseCollector):
    source_name = "institution"

    # Known AI safety / red team blogs and reports
    SOURCES = [
        {
            "name": "Gray Swan AI Blog",
            "url": "https://app.grayswan.ai/arena/blog",
            "type": "blog",
        },
        {
            "name": "Anthropic Research",
            "url": "https://www.anthropic.com/research",
            "type": "blog",
        },
        {
            "name": "MLSecOps Podcast",
            "url": "https://mlsecops.com/podcast",
            "type": "podcast",
        },
    ]

    def collect(self, query: str = "jailbreak",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        for source in self.SOURCES:
            try:
                resp = requests.get(
                    source["url"], timeout=15,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    text = self._html_to_text(resp.text)
                    if len(text) > 100:
                        items.append(RawCollectedItem(
                            text=text[:5000],
                            source=self.source_name,
                            url=source["url"],
                            title=source["name"],
                            metadata={"type": source["type"]},
                        ))
            except Exception as e:
                print(f"[Institution] Error fetching {source['name']}: {e}")
            if len(items) >= max_items:
                break
        return items[:max_items]

    @staticmethod
    def _html_to_text(html: str) -> str:
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
