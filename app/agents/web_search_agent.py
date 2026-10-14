"""DuckDuckGo web search agent with rate-limiting and network resilience."""

from typing import Dict, List
from duckduckgo_search import DDGS

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger("web_search_agent")


class WebSearchAgent:
    """Executes live web search queries with fault tolerance."""

    def __init__(self) -> None:
        self.max_results = settings.WEB_SEARCH_MAX_RESULTS
        self.timeout = settings.WEB_SEARCH_TIMEOUT_SECONDS

    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Query DuckDuckGo for top web results.
        Returns:
            List of dicts with 'title', 'url', 'body'
        """
        logger.info(f"WebSearchAgent executing query: '{query}'...")
        results: List[Dict[str, str]] = []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                # 1. Primary: Standard DDGS text search
                try:
                    for r in ddgs.text(query, max_results=self.max_results):
                        results.append({
                            "title": r.get("title", "Web Result"),
                            "url": r.get("href", ""),
                            "body": r.get("body", "")
                        })
                except Exception as sub_exc:
                    logger.warning(f"Standard DDGS query blip: {sub_exc}. Trying HTML backend...")

                # 2. Resilient fallback: Try HTML backend if default returned nothing
                if not results:
                    for r in ddgs.text(query, max_results=self.max_results, backend="html"):
                        results.append({
                            "title": r.get("title", "Web Result"),
                            "url": r.get("href", ""),
                            "body": r.get("body", "")
                        })

        except Exception as exc:
            logger.warning(f"DuckDuckGo search blip ({exc}). Trying encyclopedic fallback...")

        # 3. Knowledge Fallback: Wikipedia Search & Summary API
        if not results:
            try:
                import requests
                clean_q = query.rstrip("?").strip()
                r = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "list": "search", "srsearch": clean_q, "format": "json", "utf8": 1},
                    headers={"User-Agent": "DocumentIntelligenceAgent/1.0 (contact@internal.corp)"},
                    timeout=6
                )
                hits = r.json().get("query", {}).get("search", [])
                if hits:
                    titles = "|".join([h["title"] for h in hits[:self.max_results]])
                    r_ext = requests.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "titles": titles, "format": "json"},
                        headers={"User-Agent": "DocumentIntelligenceAgent/1.0 (contact@internal.corp)"},
                        timeout=6
                    )
                    pages = r_ext.json().get("query", {}).get("pages", {})
                    for pid, pdata in pages.items():
                        title = pdata.get("title", "")
                        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
                        extract = pdata.get("extract", "").strip()
                        if extract:
                            results.append({
                                "title": title,
                                "url": url,
                                "body": extract[:600]
                            })
            except Exception as wiki_exc:
                logger.warning(f"Wikipedia fallback failed: {wiki_exc}")

        logger.info(f"WebSearchAgent retrieved {len(results)} web result(s).")
        return results


_web_search_agent_instance = WebSearchAgent()


def get_web_search_agent() -> WebSearchAgent:
    """Retrieve singleton instance of WebSearchAgent."""
    return _web_search_agent_instance
