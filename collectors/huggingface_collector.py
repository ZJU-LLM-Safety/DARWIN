"""HuggingFace collector — search jailbreak datasets and model cards."""
import requests
from collectors.base import BaseCollector, RawCollectedItem


class HuggingFaceCollector(BaseCollector):
    source_name = "huggingface"

    KNOWN_DATASETS = [
        "walledai/WildJailbreak",
        "rubend18/ChatGPT-Jailbreak-Prompts",
    ]

    def collect(self, query: str = "jailbreak",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        items.extend(self._search_datasets(query, max_items))
        for ds in self.KNOWN_DATASETS:
            items.extend(self._scrape_dataset(ds))
            if len(items) >= max_items:
                break
        return items[:max_items]

    def _search_datasets(self, query: str, max_items: int) -> list[RawCollectedItem]:
        items = []
        try:
            url = "https://huggingface.co/api/datasets"
            params = {"search": query, "limit": min(max_items, 10), "sort": "lastModified"}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            for ds in resp.json():
                card = self._get_dataset_card(ds["id"])
                if card:
                    items.append(RawCollectedItem(
                        text=card[:5000],
                        source=self.source_name,
                        url=f"https://huggingface.co/datasets/{ds['id']}",
                        title=ds["id"],
                    ))
        except Exception as e:
            print(f"[HuggingFace] Search error: {e}")
        return items

    def _scrape_dataset(self, dataset_id: str) -> list[RawCollectedItem]:
        items = []
        try:
            # Try to get first rows via the datasets API
            url = f"https://datasets-server.huggingface.co/first-rows"
            params = {"dataset": dataset_id, "config": "default", "split": "train"}
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for row in data.get("rows", [])[:10]:
                    row_data = row.get("row", {})
                    text = str(row_data)
                    if len(text) > 50:
                        items.append(RawCollectedItem(
                            text=text[:3000],
                            source=self.source_name,
                            url=f"https://huggingface.co/datasets/{dataset_id}",
                            title=dataset_id,
                        ))
        except Exception as e:
            print(f"[HuggingFace] Scrape error for {dataset_id}: {e}")
        return items

    @staticmethod
    def _get_dataset_card(dataset_id: str) -> str:
        try:
            url = f"https://huggingface.co/api/datasets/{dataset_id}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("cardData", {}).get("description", "")
        except Exception:
            pass
        return ""
