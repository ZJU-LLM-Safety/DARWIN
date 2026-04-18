"""GitHub collector — scrape jailbreak repos and prompt collections."""
import requests
from collectors.base import BaseCollector, RawCollectedItem

# Known jailbreak repos
KNOWN_REPOS = [
    "verazuo/jailbreak_llms",
    "elder-plinius/L1B3RT45",
    "0xk1h0/ChatGPT_DAN",
]


class GitHubCollector(BaseCollector):
    source_name = "github"

    def collect(self, query: str = "LLM jailbreak prompt",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        # Search repos
        items.extend(self._search_repos(query, max_items))
        # Scrape known repos
        for repo in KNOWN_REPOS:
            items.extend(self._scrape_repo(repo, max_items=5))
            if len(items) >= max_items:
                break
        return items[:max_items]

    def _search_repos(self, query: str, max_items: int) -> list[RawCollectedItem]:
        items = []
        try:
            url = "https://api.github.com/search/repositories"
            params = {"q": query, "sort": "updated", "per_page": min(max_items, 10)}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            for repo in resp.json().get("items", []):
                readme = self._get_readme(repo["full_name"])
                if readme:
                    items.append(RawCollectedItem(
                        text=readme[:5000],
                        source=self.source_name,
                        url=repo["html_url"],
                        title=repo["full_name"],
                        metadata={"stars": repo.get("stargazers_count", 0)},
                    ))
        except Exception as e:
            print(f"[GitHub] Search error: {e}")
        return items

    def _scrape_repo(self, repo_name: str, max_items: int = 5) -> list[RawCollectedItem]:
        items = []
        try:
            readme = self._get_readme(repo_name)
            if readme:
                items.append(RawCollectedItem(
                    text=readme[:5000],
                    source=self.source_name,
                    url=f"https://github.com/{repo_name}",
                    title=repo_name,
                ))
        except Exception as e:
            print(f"[GitHub] Scrape error for {repo_name}: {e}")
        return items

    @staticmethod
    def _get_readme(repo_name: str) -> str:
        try:
            url = f"https://api.github.com/repos/{repo_name}/readme"
            resp = requests.get(url, headers={"Accept": "application/vnd.github.raw"}, timeout=15)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        return ""
