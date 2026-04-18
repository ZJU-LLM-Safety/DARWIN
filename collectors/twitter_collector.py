"""Twitter collector — search jailbreak-related tweets."""
from collectors.base import BaseCollector, RawCollectedItem


class TwitterCollector(BaseCollector):
    source_name = "twitter"

    # Known AI red team accounts to monitor
    MONITOR_ACCOUNTS = [
        "PlinythePromptr",  # Pliny the Prompter
        "alexalbert__",     # Alex Albert
    ]

    def collect(self, query: str = "LLM jailbreak prompt",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        try:
            import tweepy
            from config.settings import TWITTER_BEARER_TOKEN
            if not TWITTER_BEARER_TOKEN:
                print("[Twitter] No bearer token configured, skipping.")
                return items

            client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

            # Search recent tweets
            try:
                tweets = client.search_recent_tweets(
                    query=query, max_results=min(max_items, 100),
                    tweet_fields=["created_at", "author_id", "text"],
                )
                if tweets.data:
                    for tweet in tweets.data:
                        if len(tweet.text) > 50:
                            items.append(RawCollectedItem(
                                text=tweet.text,
                                source=self.source_name,
                                url=f"https://twitter.com/i/status/{tweet.id}",
                                title=f"Tweet {tweet.id}",
                            ))
            except Exception as e:
                print(f"[Twitter] Search error: {e}")

            # Monitor specific accounts
            for username in self.MONITOR_ACCOUNTS:
                try:
                    user = client.get_user(username=username)
                    if user.data:
                        user_tweets = client.get_users_tweets(
                            user.data.id, max_results=10,
                            tweet_fields=["created_at", "text"],
                        )
                        if user_tweets.data:
                            for tweet in user_tweets.data:
                                if len(tweet.text) > 50:
                                    items.append(RawCollectedItem(
                                        text=tweet.text,
                                        source=self.source_name,
                                        url=f"https://twitter.com/{username}/status/{tweet.id}",
                                        title=f"@{username}",
                                        metadata={"author": username},
                                    ))
                except Exception as e:
                    print(f"[Twitter] Error fetching @{username}: {e}")
                if len(items) >= max_items:
                    break
        except ImportError:
            print("[Twitter] tweepy not installed, skipping.")
        except Exception as e:
            print(f"[Twitter] Error: {e}")
        return items[:max_items]
