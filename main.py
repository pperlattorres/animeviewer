# main.py

import os
import re
import time
import asyncio
import logging
import traceback
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

import requests
import yt_dlp

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing import List
from dataclasses import asdict

from animeflv import AnimeFLV, AnimeInfo, EpisodeInfo, DownloadLinkInfo, EpisodeFormat

# ── Logging setup ─────────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, "api.log"),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])

logger = logging.getLogger("animeflv.api")
scraper_log = logging.getLogger("animeflv.scraper")

# Silenciar logs muy verbosos de librerías externas
for noisy in ("urllib3", "charset_normalizer", "cloudscraper", "httpx", "watchfiles", "watchfiles.main", "watchfiles.watcher"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="JOANG Anime API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request / Response middleware ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("→ %s %s  params=%s", request.method, request.url.path, dict(request.query_params))
    try:
        response: Response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(level, "← %s %s  status=%d  %.1fms",
                   request.method, request.url.path, response.status_code, elapsed)
        return response
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.error("← %s %s  UNHANDLED EXCEPTION  %.1fms\n%s",
                     request.method, request.url.path, elapsed, traceback.format_exc())
        raise


# ── Scraper helper ────────────────────────────────────────────────────────────
def _log_scraper_result(action: str, result, **ctx):
    """Registra el resultado de cada llamada al scraper e informa si vino vacío."""
    is_empty = isinstance(result, list) and len(result) == 0
    level = logging.WARNING if is_empty else logging.DEBUG
    ctx_str = "  ".join(f"{k}={v!r}" for k, v in ctx.items())
    scraper_log.log(level, "[%s] %s  result_type=%s  count=%s%s",
                    "EMPTY" if is_empty else "OK",
                    action,
                    type(result).__name__,
                    len(result) if isinstance(result, list) else "—",
                    f"  {ctx_str}" if ctx_str else "")


# ── yt-dlp extractor ─────────────────────────────────────────────────────────
_ydl_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ydl")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://animesflv.net/",
}

# Patterns that cover JWPlayer / Plyr / VideoJS embed pages
_VIDEO_PATTERNS = [
    # file:"https://…"  or  file: "https://…"
    r'["\']?file["\']?\s*:\s*["\']([^"\'>\s]+\.(?:m3u8|mp4|webm|mkv)(?:[^"\'>\s]*)?)["\']',
    # src="https://…mp4"
    r'src\s*=\s*["\']([^"\'>\s]+\.(?:m3u8|mp4|webm)(?:[^"\'>\s]*)?)["\']',
    # hls / hlsUrl variable
    r'(?:hls|hlsUrl|streamUrl|videoUrl)["\']?\s*[:=]\s*["\']([^"\'>\s]+)["\']',
    # Any quoted https URL with m3u8/mp4 extension (catches unpacked JWPlayer sources)
    r'["\']?(https?://[^"\'<>\s]{10,}\.(?:m3u8|mp4|webm)(?:\?[^"\'<>\s]*)?)["\']',
]

# ── Proxy SSRF guard ────────────────────────────────────────────────────────
def _proxy_url_allowed(url: str) -> bool:
    """
    Reject URLs that point to private/loopback/link-local addresses to prevent
    SSRF. Only http/https to public internet hosts are accepted.
    """
    import ipaddress
    from urllib.parse import urlparse as _urlparse
    parsed = _urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    # Block by name
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        return False
    # Block if it's an IP literal in a private/loopback/link-local range
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_unspecified:
            return False
    except ValueError:
        pass  # hostname, not an IP literal — treat as public
    return True


# Dean Edwards p,a,c,k,e,d unpacker
_PACKED_RE = re.compile(
    r"""eval\(function\(p,a,c,k,e,(?:r|d)\)\{[^}]+\}\('([\s\S]+?)',\s*(\d+),\s*\d+,\s*'([\s\S]+?)'\.split\('\|'\)""",
    re.DOTALL,
)


def _unpack_packed_js(js: str) -> str | None:
    """Unpack Dean Edwards' eval(function(p,a,c,k,e,d){...}) obfuscation."""
    m = _PACKED_RE.search(js)
    if not m:
        return None
    p_str, a_str, k_str = m.group(1), m.group(2), m.group(3)
    a = int(a_str)
    k = k_str.split('|')

    def lookup(match):
        word = match.group(0)
        try:
            idx = int(word, a)
        except ValueError:
            return word
        if 0 <= idx < len(k) and k[idx]:
            return k[idx]
        # if k[idx] is empty, keep original word
        return word

    return re.sub(r'\b[a-zA-Z0-9]+\b', lookup, p_str)


def _scrape_embed(url: str) -> dict:
    """
    Minimal fallback scraper: fetch the embed page and search for video URLs
    using common JWPlayer / Plyr patterns, plus site-specific extractors.
    Returns the same dict shape as _ydl_extract.
    """
    resp = requests.get(url, headers=_HEADERS, timeout=20, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text

    # ── Detect JS-only "Loading…" shell pages and try known CDN mirrors ───────
    # e.g. streamwish.to redirects via JS to hgplaycdn.com with the same path
    _JS_SHELL_DOMAINS = ["hgplaycdn.com", "hgplaynow.com"]
    if len(html) < 4000 and "Loading" in html and "<video" not in html.lower():
        from urllib.parse import urlparse
        parsed = urlparse(url)
        for mirror in _JS_SHELL_DOMAINS:
            mirror_url = f"https://{mirror}{parsed.path}"
            if parsed.query:
                mirror_url += f"?{parsed.query}"
            try:
                logger.debug(f"[scrape] JS-shell detected, trying mirror: {mirror_url}")
                resp2 = requests.get(mirror_url, headers=_HEADERS, timeout=20, allow_redirects=True)
                if resp2.ok and len(resp2.text) > 4000:
                    html = resp2.text
                    logger.debug(f"[scrape] mirror responded ({len(html)} chars)")
                    break
            except Exception:
                continue

    # ── Streamtape: robotlink element contains signed get_video URL ───────────
    st_m = re.search(r'id=["\']robotlink["\'][^>]*>([^<]+)<', html)
    if st_m:
        raw = st_m.group(1).strip()
        if raw.startswith("//"):
            video_url = "https:" + raw
        elif raw.startswith("/"):
            video_url = "https:/" + raw
        else:
            video_url = raw
        logger.info(f"[scrape] streamtape robotlink → {video_url[:80]}")
        return {"url": video_url, "ext": "mp4", "height": None, "title": ""}

    # ── Try to unpack Dean Edwards p,a,c,k,e,d obfuscation ───────────────────
    unpacked = _unpack_packed_js(html)
    search_targets = [html]
    if unpacked:
        logger.debug(f"[scrape] unpacked packed JS ({len(unpacked)} chars)")
        search_targets.append(unpacked)

    best_url = ""
    best_height = None
    best_ext = ""

    for target in search_targets:
        for pat in _VIDEO_PATTERNS:
            for m in re.finditer(pat, target, re.IGNORECASE):
                candidate = m.group(1)
                if candidate.startswith("//"):
                    candidate = "https:" + candidate
                if not candidate.startswith("http"):
                    continue
                ext = "m3u8" if ".m3u8" in candidate else ("mp4" if ".mp4" in candidate else "")
                snippet = target[max(0, m.start() - 100):m.end() + 100]
                height_m = re.search(r'(\d{3,4})p', snippet)
                height = int(height_m.group(1)) if height_m else 0
                if height > (best_height or 0) or not best_url:
                    best_url = candidate
                    best_height = height or None
                    best_ext = ext
        if best_url:
            break  # found in first target, no need to try unpacked

    if not best_url:
        raise ValueError(f"No video URL found in embed page: {url}")

    logger.info(f"[scrape] found {best_ext} {best_height}p → {best_url[:80]}")
    return {
        "url": best_url,
        "ext": best_ext or "mp4",
        "height": best_height,
        "title": "",
    }


def _ydl_extract(url: str) -> dict:
    """
    Run yt-dlp synchronously (called from a thread pool).
    Returns dict with at least {'url': ..., 'ext': ..., 'height': ..., 'title': ...}
    """
    _base_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Best mp4, then best HLS, then anything
        "format": "best[ext=mp4]/best[protocol^=m3u8]/best",
        "noplaylist": True,
        "socket_timeout": 20,
        "http_headers": _HEADERS,
        # Redirect yt-dlp error output to our logger so we control the noise
        "logger": type("_YdlLogger", (), {
            "debug": lambda self, m: None,
            "warning": lambda self, m: logger.debug("[yt-dlp] %s", m),
            "error": lambda self, m: logger.debug("[yt-dlp] %s", m),
        })(),
    }

    # First attempt: normal extractor lookup
    info = None
    for attempt, opts in enumerate([
        _base_opts,
        {**_base_opts, "force_generic_extractor": True},  # fallback for unsupported embeds
    ]):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            break  # success
        except (yt_dlp.utils.UnsupportedError, yt_dlp.utils.DownloadError) as e:
            if attempt == 0 and "Unsupported URL" in str(e):
                logger.warning(f"[extract] URL not recognized by yt-dlp, retrying with generic extractor: {url}")
            elif attempt == 1 and "Unsupported URL" in str(e):
                # Both yt-dlp passes failed — try manual HTML scraper
                logger.warning(f"[extract] yt-dlp generic extractor also failed, trying HTML scrape: {url}")
                return _scrape_embed(url)
            else:
                raise
        except Exception:
            raise

    direct_url = ""
    height = None
    ext = info.get("ext", "")

    if "url" in info:
        direct_url = info["url"]
        height = info.get("height")
    elif "formats" in info and info["formats"]:
        formats = info["formats"]
        # Prefer mp4 → m3u8 → anything, highest quality last
        def _rank(f):
            e = f.get("ext", "")
            p = f.get("protocol", "")
            h = f.get("height") or 0
            pref = 2 if e == "mp4" else (1 if "m3u8" in p else 0)
            return (pref, h)

        best = sorted([f for f in formats if f.get("url")], key=_rank)[-1]
        direct_url = best.get("url", "")
        height = best.get("height")
        ext = best.get("ext", ext)

    # Derive ext from URL if still unknown
    if not ext or ext == "unknown_video":
        if direct_url.startswith("data:video/mp4") or direct_url.startswith("data:video/mp4"):
            ext = "mp4"
        elif direct_url.startswith("data:video/"):
            # data:video/webm;base64,... → webm
            ext = direct_url.split("data:video/")[1].split(";")[0]
        elif ".m3u8" in direct_url:
            ext = "m3u8"
        elif ".mp4" in direct_url:
            ext = "mp4"

    return {
        "url": direct_url,
        "ext": ext or "mp4",
        "height": height,
        "title": info.get("title", ""),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def serialize(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [serialize(i) for i in obj]
    return obj


@app.get("/search", response_model=List[AnimeInfo])
async def search(query: str, page: int = 1):
    scraper_log.info("[CALL] search  query=%r  page=%d", query, page)
    try:
        with AnimeFLV() as client:
            results = client.search(query=query, page=page)
        _log_scraper_result("search", results, query=query, page=page)
        return results
    except Exception as e:
        scraper_log.error("[ERROR] search  query=%r  page=%d\n%s", query, page, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anime/{anime_id}", response_model=AnimeInfo)
async def get_anime_info(anime_id: str):
    scraper_log.info("[CALL] get_anime_info  anime_id=%r", anime_id)
    try:
        with AnimeFLV() as client:
            info = client.get_anime_info(anime_id)
        episode_count = len(info.episodes) if info.episodes else 0
        scraper_log.debug("[OK] get_anime_info  anime_id=%r  title=%r  episodes=%d",
                          anime_id, info.title, episode_count)
        if episode_count == 0:
            scraper_log.warning("[EMPTY] get_anime_info returned 0 episodes for anime_id=%r", anime_id)
        return info
    except Exception as e:
        scraper_log.error("[ERROR] get_anime_info  anime_id=%r\n%s", anime_id, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anime/{anime_id}/episodes", response_model=List[EpisodeInfo])
async def get_anime_episodes(anime_id: str):
    scraper_log.info("[CALL] get_anime_episodes  anime_id=%r", anime_id)
    try:
        with AnimeFLV() as client:
            info = client.get_anime_info(anime_id)
        episodes = info.episodes or []
        _log_scraper_result("get_anime_episodes", episodes, anime_id=anime_id)
        return episodes
    except Exception as e:
        scraper_log.error("[ERROR] get_anime_episodes  anime_id=%r\n%s", anime_id, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anime/{anime_id}/episode/{episode}/links", response_model=List[DownloadLinkInfo])
async def get_download_links(anime_id: str, episode: int, dubbed: bool = False):
    fmt = EpisodeFormat.Dubbed if dubbed else EpisodeFormat.Subtitled
    scraper_log.info("[CALL] get_links  anime_id=%r  episode=%d  format=%s", anime_id, episode, fmt.name)
    try:
        with AnimeFLV() as client:
            links = client.get_links(anime_id, episode, format=fmt)
        _log_scraper_result("get_links", links, anime_id=anime_id, episode=episode, format=fmt.name)
        return links
    except Exception as e:
        scraper_log.error("[ERROR] get_links  anime_id=%r  episode=%d\n%s", anime_id, episode, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anime/{anime_id}/episode/{episode}/servers")
async def get_video_servers(anime_id: str, episode: int, dubbed: bool = False):
    fmt = EpisodeFormat.Dubbed if dubbed else EpisodeFormat.Subtitled
    scraper_log.info("[CALL] get_video_servers  anime_id=%r  episode=%d  format=%s", anime_id, episode, fmt.name)
    try:
        with AnimeFLV() as client:
            servers = client.get_video_servers(anime_id, episode, format=fmt)
        flat = [s for group in servers for s in (group if isinstance(group, list) else [group])]
        _log_scraper_result("get_video_servers", flat, anime_id=anime_id, episode=episode, format=fmt.name)
        return servers
    except Exception as e:
        scraper_log.error("[ERROR] get_video_servers  anime_id=%r  episode=%d\n%s", anime_id, episode, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latest/episodes")
async def get_latest_episodes():
    scraper_log.info("[CALL] get_latest_episodes")
    try:
        with AnimeFLV() as client:
            items = client.get_latest_episodes()
        _log_scraper_result("get_latest_episodes", items)
        return serialize(items)
    except Exception as e:
        scraper_log.error("[ERROR] get_latest_episodes\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latest/animes")
async def get_latest_animes():
    scraper_log.info("[CALL] get_latest_animes")
    try:
        with AnimeFLV() as client:
            items = client.get_latest_animes()
        _log_scraper_result("get_latest_animes", items)
        return serialize(items)
    except Exception as e:
        scraper_log.error("[ERROR] get_latest_animes\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/extract")
async def extract_direct_url(url: str):
    """
    Use yt-dlp to extract a direct playable video URL from a player/embed page.
    Returns { url, ext, height, title } or 422/408 on failure.
    """
    logger.info("[extract] url=%r", url)
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_ydl_executor, _ydl_extract, url),
            timeout=30,
        )
    except asyncio.TimeoutError:
        logger.warning("[extract] Timeout for url=%r", url)
        raise HTTPException(status_code=408, detail="Extraction timed out after 30s")
    except (ValueError, yt_dlp.utils.DownloadError, yt_dlp.utils.UnsupportedError) as e:
        # Expected: unsupported site or no video found in page — not a code bug
        logger.warning("[extract] No extractable URL for %r — %s", url, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("[extract] Unexpected error for url=%r\n%s", url, traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e))

    if not result.get("url"):
        raise HTTPException(status_code=404, detail="No playable URL found")

    logger.info("[extract] OK ext=%s height=%s url=%r", result["ext"], result["height"], result["url"][:80])
    return result


@app.get("/proxy")
async def proxy_media(url: str, request: Request):
    """
    Server-side proxy for media URLs that have CORS restrictions.
    For m3u8 playlists it rewrites segment URLs so they also go through this proxy.
    """
    from urllib.parse import urljoin, urlparse, urlunparse, quote
    if not _proxy_url_allowed(url):
        raise HTTPException(status_code=403, detail="URL not allowed: only public HTTP(S) hosts are accepted")

    # Build referer from origin domain
    parsed = urlparse(url)
    referer = f"{parsed.scheme}://{parsed.netloc}/"
    proxy_headers = {**_HEADERS, "Referer": referer, "Origin": referer}

    try:
        resp = requests.get(url, headers=proxy_headers, timeout=20, stream=True)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("[proxy] fetch failed for %r: %s", url[:80], e)
        raise HTTPException(status_code=502, detail=str(e))

    content_type = resp.headers.get("content-type", "application/octet-stream")

    # ── Rewrite m3u8 playlists ─────────────────────────────────────────────
    if "mpegurl" in content_type or url.split("?")[0].endswith(".m3u8"):
        base_url = url
        lines = []
        for line in resp.text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                # Resolve relative segment/playlist URL → absolute → through proxy
                abs_seg = urljoin(base_url, stripped)
                lines.append("/proxy?url=" + quote(abs_seg, safe=""))
            else:
                lines.append(line)
        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="application/vnd.apple.mpegurl",
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"},
        )

    # ── Stream everything else (segments, mp4…) ────────────────────────────
    def _iter():
        for chunk in resp.iter_content(chunk_size=65536):
            yield chunk

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache",
        "Content-Type": content_type,
    }
    if "content-length" in resp.headers:
        headers["Content-Length"] = resp.headers["content-length"]
    if "content-range" in resp.headers:
        headers["Content-Range"] = resp.headers["content-range"]

    return StreamingResponse(_iter(), status_code=resp.status_code,
                             media_type=content_type, headers=headers)


# ── Punto de arranque ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        reload_dirs=[os.path.dirname(__file__)],
        reload_excludes=["logs", "logs/*", "*.log"],
    )
