"""Arxiv collector — search for jailbreak papers."""
import requests
import xml.etree.ElementTree as ET
from collectors.base import BaseCollector, RawCollectedItem


class ArxivCollector(BaseCollector):
    source_name = "arxiv"

    def collect(self, query: str = "LLM jailbreak attack",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_items,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()

            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)

                title_text = title.text.strip() if title is not None else ""
                summary_text = summary.text.strip() if summary is not None else ""
                link_text = link.text.strip() if link is not None else ""

                if summary_text:
                    items.append(RawCollectedItem(
                        text=f"{title_text}\n\n{summary_text}",
                        source=self.source_name,
                        url=link_text,
                        title=title_text,
                    ))
        except Exception as e:
            print(f"[Arxiv] Error: {e}")
        return items[:max_items]
