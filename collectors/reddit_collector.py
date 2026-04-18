"""Reddit collector — search jailbreak-related posts."""
from collectors.base import BaseCollector, RawCollectedItem


class RedditCollector(BaseCollector):
    source_name = "reddit"

    SUBREDDITS = ["ChatGPT", "ChatGPTJailbreak", "LocalLLaMA", "artificial"]

    def collect(self, query: str = "LLM jailbreak prompt",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        try:
            import praw
            from config.settings import (
                REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
            )
            if not REDDIT_CLIENT_ID:
                print("[Reddit] No credentials configured, skipping.")
                return items

            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            for sub_name in self.SUBREDDITS:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for post in subreddit.search(query, limit=max_items // len(self.SUBREDDITS)):
                        text = f"{post.title}\n\n{post.selftext}"
                        if len(text.strip()) > 50:
                            items.append(RawCollectedItem(
                                text=text[:5000],
                                source=self.source_name,
                                url=f"https://reddit.com{post.permalink}",
                                title=post.title,
                                metadata={"subreddit": sub_name, "score": post.score},
                            ))
                except Exception as e:
                    print(f"[Reddit] Error in r/{sub_name}: {e}")
                if len(items) >= max_items:
                    break
        except ImportError:
            print("[Reddit] praw not installed, skipping.")
        except Exception as e:
            print(f"[Reddit] Error: {e}")
        return items[:max_items]
