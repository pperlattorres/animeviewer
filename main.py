# main.py

import os
import time
import logging
import traceback
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
