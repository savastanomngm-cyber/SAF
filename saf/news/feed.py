"""News fetching with Shadow Alpha relevance scoring.
Headlines are scored against SHADOW_KEYWORDS; only score >= 2.0
reaches analyst prompts. Everything else stays browsable."""
import requests
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime

# ─── Shadow Alpha relevance keywords (from improvements.txt Part 11) ───
SHADOW_KEYWORDS = {
    # Supply chain bottlenecks
    "bottleneck": 2.0,
    "supply shortage": 2.0,
    "supply chain": 1.5,
    "lead time": 1.5,
    "capacity sold out": 2.0,
    "sole supplier": 2.5,
    "sole source": 2.5,
    "single source": 2.0,
    "export control": 2.0,
    "export ban": 2.0,
    "force majeure": 2.5,
    "backlog": 1.5,
    "price increase": 1.5,
    "pricing power": 2.0,
    "monopoly": 2.0,
    "oligopoly": 2.0,
    "duopoly": 2.5,
    "market share": 1.0,
    "dominant supplier": 2.0,
    # Semiconductor specific
    "cowos": 2.5,
    "hbm": 2.0,
    "hbm3e": 2.5,
    "abf substrate": 2.5,
    "euv": 2.0,
    "lithography": 1.5,
    "foundry": 1.0,
    "advanced packaging": 2.0,
    "chip shortage": 2.0,
    "semiconductor equipment": 1.5,
    # Geopolitical / sanctions
    "sanctions": 1.5,
    "sanctioned": 1.5,
    "tariff": 1.0,
    "trade war": 1.0,
    "export restriction": 2.0,
    "entity list": 2.0,
    "arms embargo": 2.0,
    "dual-use": 1.5,
    "technology transfer": 1.5,
    # Physical constraints
    "transformer shortage": 2.5,
    "transformer lead time": 2.0,
    "grid capacity": 1.5,
    "helium": 2.0,
    "helium-3": 2.5,
    "helium shortage": 2.5,
    "cryogenic": 1.5,
    "cdmo capacity": 2.0,
    "bioreactor": 2.0,
    "vial shortage": 2.5,
    "borosilicate": 2.5,
    "pharma glass": 2.5,
    "water scarcity": 2.0,
    "desalination": 1.5,
    "rare earth": 2.0,
    "lithium": 1.5,
    "cobalt": 1.5,
    "boron": 2.0,
    # Insider / institutional signals
    "13f": 1.0,
    "congressional trading": 1.5,
    "pelosi": 1.0,
    "insider buying": 1.5,
    "insider selling": 1.0,
    "institutional ownership": 1.0,
    "short squeeze": 1.0,
    "short interest": 1.0,
    # Energy / commodities
    "freight rate": 1.5,
    "shipping rate": 1.5,
    "tanker": 1.0,
    "commodity supercycle": 1.5,
    "capex cycle": 1.5,
    "refinery": 1.0,
    "pipeline capacity": 1.5,
    "lng": 1.5,
    # Defense
    "defense spending": 1.5,
    "defense budget": 1.5,
    "military contract": 1.5,
    "drone": 1.0,
    "counter-drone": 1.5,
    "c-uas": 2.0,
    "surveillance": 1.0,
    # Quantum
    "quantum computing": 1.5,
    "quantum error correction": 2.0,
    "photonic": 1.5,
    "cryostat": 2.0,
    "dilution refrigerator": 2.5,
}

SIGNAL_THRESHOLD = 2.0  # Only headlines scoring >= this reach analyst prompts


def relevance_score(headline: str) -> float:
    """Score a headline against Shadow Alpha keywords."""
    text = headline.lower()
    return sum(w for k, w in SHADOW_KEYWORDS.items() if k in text)


def extract_keywords(headline: str) -> list:
    """Return matched keywords for display."""
    text = headline.lower()
    return [k for k in SHADOW_KEYWORDS if k in text]


def fetch_news_adhoc(query: str, limit: int = 12) -> list:
    """Fetch news from Google News RSS, scored and sorted by relevance."""
    enriched = query
    # Disambiguate ticker-only queries (CAT -> Caterpillar)
    if query.isupper() and len(query) <= 5 and query.isalpha():
        try:
            from .. import store
            fund = store.get_fundamentals(query, max_age_days=30)
            if fund and fund.get("longName"):
                enriched = f"{fund['longName']} {query} stock"
            elif fund and fund.get("sector"):
                enriched = f"{query} {fund['sector']} stock"
        except Exception:
            pass

    url = f"https://news.google.com/rss/search?q={quote(enriched)}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "SkiaAlpha/4.2 research"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            source_el = item.find("source")
            source = source_el.text if source_el is not None else "Unknown"

            score = relevance_score(title)
            keywords = extract_keywords(title)

            items.append({
                "title": title,
                "link": link,
                "source": source,
                "pub_date": pub_date,
                "relevance": round(score, 1),
                "keywords": keywords,
                "is_signal": score >= SIGNAL_THRESHOLD,
            })

        # Sort by relevance descending, then by date
        items.sort(key=lambda x: (-x["relevance"], x["pub_date"]), reverse=False)
        items.sort(key=lambda x: x["relevance"], reverse=True)

        return items[:limit]
    except Exception as e:
        print(f"[news] fetch error for '{query}': {e}")
        return []


def fetch_news(tickers: list, limit_per_ticker: int = 3) -> list:
    """Fetch news for multiple tickers (used for dashboard/news tab)."""
    all_items = []
    for t in tickers[:10]:  # Cap to avoid rate limiting
        items = fetch_news_adhoc(t, limit=limit_per_ticker)
        all_items.extend(items)
    all_items.sort(key=lambda x: x["relevance"], reverse=True)
    return all_items


def filter_for_analyst(items: list) -> list:
    """Only return items that meet the signal threshold for LLM prompts."""
    return [i for i in items if i["relevance"] >= SIGNAL_THRESHOLD]