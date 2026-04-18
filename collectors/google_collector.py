"""Google search collector — find jailbreak-related web pages."""
import requests
from collectors.base import BaseCollector, RawCollectedItem

# Known jailbreak resource URLs
KNOWN_URLS = [
    "https://www.promptfoo.dev/blog/how-to-jailbreak-llms/",
]


class GoogleCollector(BaseCollector):
    source_name = "google"

    def collect(self, query: str = "LLM jailbreak prompt techniques",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        # Scrape known URLs
        for url in KNOWN_URLS:
            item = self._scrape_url(url)
            if item:
                items.append(item)

        # Simple web search via requests (no API key needed)
        items.extend(self._search_web(query, max_items))
        return items[:max_items]

    def _search_web(self, query: str, max_items: int) -> list[RawCollectedItem]:
        """Use DuckDuckGo HTML search as a free alternative."""
        items = []
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = requests.post(url, data={"q": query}, timeout=30,
                                 headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                # Simple extraction of result snippets
                from html.parser import HTMLParser

                class ResultParser(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.results = []
                        self._in_result = False
                        self._current = ""
                        self._current_url = ""

                    def handle_starttag(self, tag, attrs):
                        attrs_dict = dict(attrs)
                        if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                            self._in_result = True
                            self._current_url = attrs_dict.get("href", "")

                    def handle_data(self, data):
                        if self._in_result:
                            self._current += data

                    def handle_endtag(self, tag):
                        if tag == "a" and self._in_result:
                            self._in_result = False
                            if self._current.strip():
                                self.results.append((self._current.strip(), self._current_url))
                            self._current = ""

                parser = ResultParser()
                parser.feed(resp.text)
                for title, result_url in parser.results[:max_items]:
                    scraped = self._scrape_url(result_url)
                    if scraped:
                        items.append(scraped)
        except Exception as e:
            print(f"[Google] Search error: {e}")
        return items

    def _scrape_url(self, url: str) -> RawCollectedItem | None:
        try:
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                # Extract text content (simple approach)
                text = self._html_to_text(resp.text)
                if len(text) > 100:
                    return RawCollectedItem(
                        text=text[:5000],
                        source=self.source_name,
                        url=url,
                        title=url.split("/")[-1] or url,
                    )
        except Exception as e:
            print(f"[Google] Scrape error for {url}: {e}")
        return None

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Very basic HTML to text conversion."""
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
