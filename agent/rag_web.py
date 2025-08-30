#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, hashlib
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse

CACHE_DIR = Path(__file__).resolve().parents[1]/".cache"/"web_rag"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def is_allowed(url: str, allow_domains: List[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    # strip leading 'www.'
    if host.startswith("www."):
        host = host[4:]
    for dom in allow_domains or []:
        dom = (dom or "").lower().lstrip(".")
        if not dom:
            continue
        # allow subdomains: endswith match on hostname
        if host == dom or host.endswith("." + dom):
            return True
    return False

def cache_get(url: str) -> str | None:
    p = CACHE_DIR/(f"{_h(url)}.json")
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("text")
    except Exception:
        return None

def cache_put(url: str, text: str) -> None:
    p = CACHE_DIR/(f"{_h(url)}.json")
    payload = {"url": url, "ts": time.time(), "text": text}
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def search_bing(query: str, api_key: str, max_results: int = 5) -> List[Dict[str, str]]:
    import requests
    url = "https://api.bing.microsoft.com/v7.0/search"
    resp = requests.get(
        url,
        params={"q": query, "count": max_results},
        headers={"Ocp-Apim-Subscription-Key": api_key, "User-Agent": "VIRTK-agent/1.0"},
        timeout=8,
    )
    resp.raise_for_status()
    items = resp.json().get("webPages", {}).get("value", [])
    return [{"name": x.get("name", ""), "url": x.get("url", "")} for x in items if x.get("url")]

def search_google(query: str, api_key: str, cx: str, max_results: int = 5) -> List[Dict[str, str]]:
    import requests
    # Google Programmable Search Engine (Custom Search JSON API)
    # Requires GOOGLE_API_KEY and GOOGLE_CSE_ID (cx)
    url = "https://www.googleapis.com/customsearch/v1"
    num = max(1, min(10, max_results))  # API allows up to 10 per request
    resp = requests.get(
        url,
        params={"q": query, "key": api_key, "cx": cx, "num": num},
        headers={"User-Agent": "VIRTK-agent/1.0"},
        timeout=8,
    )
    resp.raise_for_status()
    items = resp.json().get("items", []) or []
    results = []
    for it in items:
        link = it.get("link")
        if link:
            results.append({"name": it.get("title", ""), "url": link})
    return results

def fetch_and_extract(url: str, timeout_s: int = 8) -> str:
    import requests
    from bs4 import BeautifulSoup
    cached = cache_get(url)
    if cached is not None:
        return cached
    r = requests.get(url, headers={"User-Agent": "VIRTK-agent/1.0"}, timeout=timeout_s)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = " ".join(soup.stripped_strings)
    text = text[:20000]
    cache_put(url, text)
    return text

def web_retrieve(query: str, provider: str, allow_domains: List[str], max_results: int, timeout_s: int) -> List[str]:
    provider = (provider or "bing").lower()
    blobs: List[str] = []
    try:
        if provider == "google":
            g_key = os.getenv("GOOGLE_API_KEY", "").strip()
            g_cx = os.getenv("GOOGLE_CSE_ID", "").strip()
            if not g_key or not g_cx:
                return []
            results = search_google(query, g_key, g_cx, max_results=max_results)
        elif provider == "bing":
            b_key = os.getenv("BING_API_KEY", "").strip()
            if not b_key:
                return []
            results = search_bing(query, b_key, max_results=max_results)
        else:
            return []
    except Exception:
        return []

    # domain allowlist filter
    urls = [r["url"] for r in results if is_allowed(r.get("url", ""), allow_domains)]
    for u in urls:
        try:
            blobs.append(fetch_and_extract(u, timeout_s=timeout_s))
        except Exception:
            continue
    return blobs
