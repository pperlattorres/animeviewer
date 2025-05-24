# main.py

from fastapi import FastAPI, HTTPException
from typing import List
from dataclasses import asdict

from animeflv import AnimeFLV, AnimeInfo, EpisodeInfo, DownloadLinkInfo, EpisodeFormat

app = FastAPI(title="AnimeFLV Scraper API", version="0.1")

def serialize(obj):
    """Convierte dataclass en dict recursivo"""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [serialize(i) for i in obj]
    return obj

@app.get("/search", response_model=List[AnimeInfo])
async def search(query: str, page: int = 1):
    try:
        with AnimeFLV() as client:
            results = client.search(query=query, page=page)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anime/{anime_id}", response_model=AnimeInfo)
async def get_anime_info(anime_id: str):
    try:
        with AnimeFLV() as client:
            info = client.get_anime_info(anime_id)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anime/{anime_id}/episodes", response_model=List[EpisodeInfo])
async def get_latest_episodes(anime_id: str):
    try:
        with AnimeFLV() as client:
            # Si tu clase tuviera método específico, ajústalo; 
            # aquí usamos list() como alias de search sin query
            eps = client.get_latest_episodes()
        return eps
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anime/{anime_id}/episode/{episode}/links", response_model=List[DownloadLinkInfo])
async def get_download_links(anime_id: str, episode: int, dubbed: bool = False):
    try:
        fmt = EpisodeFormat.Dubbed if dubbed else EpisodeFormat.Subtitled
        with AnimeFLV() as client:
            links = client.get_links(anime_id, episode, format=fmt)
        return links
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anime/{anime_id}/episode/{episode}/servers")
async def get_video_servers(anime_id: str, episode: int, dubbed: bool = False):
    try:
        fmt = EpisodeFormat.Dubbed if dubbed else EpisodeFormat.Subtitled
        with AnimeFLV() as client:
            servers = client.get_video_servers(anime_id, episode, format=fmt)
        return servers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Punto de arranque
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
