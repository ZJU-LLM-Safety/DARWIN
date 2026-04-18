"""External evolution — collect → extract → dedup → sandbox → pool."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

from config.prompts import EXTERNAL_STRATEGY_REVIEW_PROMPT
from config.settings import EXTERNAL_REVIEW_DIR, EXTERNAL_SANDBOX_ENABLED
from models.llm_manager import LLMManager
from collectors.base import RawCollectedItem
from collectors.github_collector import GitHubCollector
from collectors.huggingface_collector import HuggingFaceCollector
from collectors.reddit_collector import RedditCollector
from collectors.google_collector import GoogleCollector
from collectors.twitter_collector import TwitterCollector
from collectors.discord_collector import DiscordCollector
from collectors.arxiv_collector import ArxivCollector
from collectors.institution_collector import InstitutionCollector
from scripts.extract_external_strategies import (
    DEFAULT_CONFIG_PATH as EXTERNAL_EXTRACT_CONFIG_PATH,
    SourceItem as ExtractSourceItem,
    build_client as build_extract_client,
    extract_strategy_card,
    parse_server_config as parse_extract_server_config,
)


class ExternalEvolution:
    """Orchestrate external data collection → strategy extraction → pool insertion."""

    def __init__(self, sqlite=None, chroma=None, *, sandbox_enabled: bool = EXTERNAL_SANDBOX_ENABLED):
        self.pool = None
        self.mgr = LLMManager()
        self.sandbox = None
        self.sandbox_enabled = sandbox_enabled
        self._extract_client = None
        self.collectors = [
            GitHubCollector(),
            HuggingFaceCollector(),
            RedditCollector(),
            GoogleCollector(),
            TwitterCollector(),
            DiscordCollector(),
            ArxivCollector(),
            InstitutionCollector(),
        ]
        if sqlite is not None and chroma is not None:
            from strategy.strategy_pool import StrategyPool
            from sandbox.validator import SandboxValidator

            self.pool = StrategyPool(sqlite, chroma)
            self.sandbox = SandboxValidator()

    def run(self, max_items_per_source: int = 20, verbose: bool = True) -> dict:
        """Full external evolution pipeline. Returns stats."""
        if self.pool is None or self.sandbox is None:
            raise RuntimeError("ExternalEvolution.run requires sqlite and chroma components.")

        stats = {
            "collected": 0, "viable": 0, "duplicates": 0,
            "sandbox_passed": 0, "added": 0,
        }

        # Step 1: Collect from all sources
        all_items = self._collect_all_items(
            max_items_per_source=max_items_per_source,
            verbose=verbose,
        )

        stats["collected"] = len(all_items)
        if verbose:
            print(f"[External] Total collected: {stats['collected']}")

        # Step 2: Extract legacy-style strategies via the gpt-5.4 extractor.
        for item in all_items:
            result = self._extract_strategy_template(item)
            if result is None:
                continue

            stats["viable"] += 1
            strategy_name, template_text, tags = result

            # Step 3: Dedup check
            if self.pool.vec.is_duplicate_strategy(template_text):
                stats["duplicates"] += 1
                continue

            sandbox_success_rate = 0.0
            sandbox_avg_score = 0.0
            if self.sandbox_enabled:
                validation = self.sandbox.validate_detailed(
                    template_text,
                    strategy_name=strategy_name,
                    strategy_id=f"external_{item.source}_{item.metadata.get('row_index', item.title or 'item')}",
                    source_group=f"external_{item.source}",
                    source_path=item.url,
                    verbose=verbose,
                )
                if not validation["passed"]:
                    continue
                sandbox_success_rate = validation["success_rate"]
                sandbox_avg_score = validation["avg_score"]
                stats["sandbox_passed"] += 1

            # Step 5: Add to pool
            sid = self.pool.add_strategy(
                name=strategy_name,
                text=template_text,
                tags=tags,
                source=f"external_{item.source}",
                source_group=f"external_{item.source}",
                source_path=item.url,
                metadata={
                    "title": item.title,
                    "url": item.url,
                    "collected_at": item.collected_at,
                    "collector_metadata": item.metadata,
                },
                sandbox_success_rate=sandbox_success_rate,
                sandbox_avg_score=sandbox_avg_score,
            )
            if sid:
                stats["added"] += 1
                if verbose:
                    print(f"  [External] Added strategy (id={sid}) from {item.source}")

        if verbose:
            print(f"[External] Done. Stats: {stats}")
        return stats

    def review(self, max_items_per_source: int = 20, verbose: bool = True,
               output_path: str | None = None) -> dict:
        """Collect sources and normalize them into safe, reviewable strategy cards."""
        os.makedirs(EXTERNAL_REVIEW_DIR, exist_ok=True)
        all_items = self._collect_all_items(
            max_items_per_source=max_items_per_source,
            verbose=verbose,
        )

        cards = []
        for idx, item in enumerate(all_items, start=1):
            if verbose:
                print(f"[Review] Analyzing {idx}/{len(all_items)} from {item.source}...")
            card = self._review_item(item)
            if card is None:
                continue
            cards.append(card)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path or os.path.join(
            EXTERNAL_REVIEW_DIR, f"external_strategy_review_{timestamp}.json"
        )
        md_path = os.path.splitext(json_path)[0] + ".md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._render_review_markdown(cards))

        stats = {
            "collected": len(all_items),
            "reviewed": len(cards),
            "json_path": json_path,
            "markdown_path": md_path,
        }
        if verbose:
            print(f"[Review] Done. Stats: {stats}")
        return stats

    def _collect_all_items(self, max_items_per_source: int, verbose: bool) -> list[RawCollectedItem]:
        """Collect raw items from all configured sources."""
        all_items: list[RawCollectedItem] = []
        for collector in self.collectors:
            if verbose:
                print(f"[External] Collecting from {collector.source_name}...")
            try:
                items = collector.collect(max_items=max_items_per_source)
                all_items.extend(items)
                if verbose:
                    print(f"  → {len(items)} items")
            except Exception as e:
                print(f"  → Error: {e}")
        return all_items

    def _extract_strategy_template(self, item: RawCollectedItem) -> tuple[str, str, str] | None:
        """Use the gpt-5.4 extraction script logic to build a DARWIN-style template."""
        try:
            client = self._get_extract_client()
            source_item = ExtractSourceItem(
                item_id=str(item.metadata.get("row_index", item.title or item.url or item.source)),
                text=item.text,
                metadata=item.metadata,
            )
            card = extract_strategy_card(client, "gpt-5.4", source_item, item.source)
        except Exception:
            return None

        if not self._looks_like_template(card):
            return None

        name = self._extract_template_name(card) or f"external_{item.source}_strategy"
        tags = ",".join(filter(None, [item.source, "external"]))
        return name, card, tags

    def _get_extract_client(self):
        if self._extract_client is None:
            config = parse_extract_server_config(EXTERNAL_EXTRACT_CONFIG_PATH)
            self._extract_client = build_extract_client(config)
        return self._extract_client

    @staticmethod
    def _extract_template_name(template_text: str) -> str:
        match = re.search(r"\[\[(.*?)\]\]", template_text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _looks_like_template(text: str) -> bool:
        required = [
            "{harmful_prompt}",
            "Instructions:",
            "Your Task:",
            "Output Requirements:",
        ]
        return all(token in text for token in required)

    def _review_item(self, item: RawCollectedItem) -> dict | None:
        """Convert a raw source item into a sanitized strategy card."""
        prompt = EXTERNAL_STRATEGY_REVIEW_PROMPT.format(raw_text=item.text[:4000])
        output = self.mgr.api_generate(
            prompt,
            temperature=0.2,
            max_tokens=900,
        )
        if not output:
            return None

        parsed = self._parse_review_output(output)
        if parsed.get("VIABLE") != "YES":
            return None

        instructions = self._split_pipe_list(parsed.get("INSTRUCTIONS", ""))
        tags = [tag.strip() for tag in parsed.get("TAGS", "").split(",") if tag.strip()]
        cues = [cue.strip() for cue in parsed.get("DISTINCTIVE_CUES", "").split(",") if cue.strip()]

        return {
            "source": item.source,
            "title": item.title,
            "url": item.url,
            "metadata": item.metadata,
            "name": parsed.get("NAME", ""),
            "primary_goal": parsed.get("PRIMARY_GOAL", ""),
            "mechanism": parsed.get("MECHANISM", ""),
            "instructions": instructions,
            "distinctive_cues": cues,
            "nearest_seed": parsed.get("NEAREST_SEED", ""),
            "alignment": parsed.get("ALIGNMENT", ""),
            "safe_summary": parsed.get("SAFE_SUMMARY", ""),
            "tags": tags,
            "reviewed_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _parse_review_output(output: str) -> dict[str, str]:
        """Parse the line-oriented strategy review format."""
        parsed: dict[str, str] = {}
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip().upper()] = value.strip()
        return parsed

    @staticmethod
    def _split_pipe_list(value: str) -> list[str]:
        return [part.strip() for part in value.split("||") if part.strip()]

    @staticmethod
    def _render_review_markdown(cards: list[dict]) -> str:
        """Render a compact markdown report for manual review."""
        lines = [
            "# External Strategy Review",
            "",
            "This report contains sanitized, high-level strategy cards only.",
            "",
        ]
        for idx, card in enumerate(cards, start=1):
            lines.extend([
                f"## {idx}. {card['name'] or 'Unnamed Strategy'}",
                "",
                f"- Source: {card['source']}",
                f"- Title: {card['title'] or 'N/A'}",
                f"- URL: {card['url'] or 'N/A'}",
                f"- Nearest seed: {card['nearest_seed'] or 'NONE'}",
                f"- Tags: {', '.join(card['tags']) if card['tags'] else 'N/A'}",
                f"- Distinctive cues: {', '.join(card['distinctive_cues']) if card['distinctive_cues'] else 'N/A'}",
                "",
                f"Primary goal: {card['primary_goal']}",
                "",
                f"Mechanism: {card['mechanism']}",
                "",
                "Abstract instructions:",
            ])
            for instruction in card["instructions"]:
                lines.append(f"- {instruction}")
            lines.extend([
                "",
                f"Alignment to legacy seed: {card['alignment']}",
                "",
                f"Safe summary: {card['safe_summary']}",
                "",
            ])
        return "\n".join(lines)
