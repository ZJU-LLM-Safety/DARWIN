"""Discord collector — scrape jailbreak channels."""
from collectors.base import BaseCollector, RawCollectedItem


class DiscordCollector(BaseCollector):
    source_name = "discord"

    # BASI community channel
    CHANNELS = [
        {"guild_id": 1105891499641684019, "channel_id": 1432845259825741824},
    ]

    def collect(self, query: str = "jailbreak",
                max_items: int = 20) -> list[RawCollectedItem]:
        items = []
        try:
            import discord
            import asyncio
            from config.settings import DISCORD_BOT_TOKEN
            if not DISCORD_BOT_TOKEN:
                print("[Discord] No bot token configured, skipping.")
                return items

            items = asyncio.get_event_loop().run_until_complete(
                self._async_collect(DISCORD_BOT_TOKEN, query, max_items)
            )
        except ImportError:
            print("[Discord] discord.py not installed, skipping.")
        except RuntimeError:
            # Event loop already running
            import asyncio
            loop = asyncio.new_event_loop()
            items = loop.run_until_complete(
                self._async_collect(
                    __import__("config.settings", fromlist=["DISCORD_BOT_TOKEN"]).DISCORD_BOT_TOKEN,
                    query, max_items,
                )
            )
            loop.close()
        except Exception as e:
            print(f"[Discord] Error: {e}")
        return items[:max_items]

    async def _async_collect(self, token: str, query: str,
                             max_items: int) -> list[RawCollectedItem]:
        import discord
        items = []
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            for ch_info in self.CHANNELS:
                try:
                    channel = client.get_channel(ch_info["channel_id"])
                    if channel is None:
                        channel = await client.fetch_channel(ch_info["channel_id"])
                    async for message in channel.history(limit=max_items * 2):
                        if query.lower() in message.content.lower() or len(message.content) > 100:
                            items.append(RawCollectedItem(
                                text=message.content[:3000],
                                source=self.source_name,
                                url=f"https://discord.com/channels/{ch_info['guild_id']}/{ch_info['channel_id']}/{message.id}",
                                title=f"Discord #{channel.name}",
                                metadata={"author": str(message.author)},
                            ))
                        if len(items) >= max_items:
                            break
                except Exception as e:
                    print(f"[Discord] Channel error: {e}")
            await client.close()

        try:
            await asyncio.wait_for(client.start(token), timeout=30)
        except asyncio.TimeoutError:
            await client.close()
        return items
