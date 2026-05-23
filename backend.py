"""
LYRIMOD — Music Emotion Analyzer Backend
Run with: uvicorn backend:app --reload --port 8000
"""

import asyncio
import json
import os
import re
import tempfile
from typing import AsyncIterator, Optional

import acoustid
import lyricsgenius
import requests
from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mutagen import File as MutagenFile
from pydantic import BaseModel
from transformers import pipeline
from urllib.parse import quote

# ── App setup ────────────────────────────────────────────────
app = FastAPI(title="LYRIMOD API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ML Model ─────────────────────────────────────────────────
print("Loading Hartmann emotion model...")
# hartmann = pipeline("text-classification", model="./hartmann_model", top_k=None)
hartmann = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
print("Model ready.")

# ── Config ───────────────────────────────────────────────────
ACOUSTID_KEY  = "k07QeKauYO"
GENIUS_TOKEN  = "Y7kfp7zUJqP-AqFAQwpjKqCM6ugvZXeVPfucmKOa98ma2pJuG2Xb1llpZJdrIPPD"

genius = lyricsgenius.Genius(GENIUS_TOKEN, remove_section_headers=True)

EMOTION_TO_SCALE = {
    "joy": 5, "surprise": 4, "neutral": 3,
    "sadness": 2, "fear": 1, "disgust": 1, "anger": 1,
}

AUDIO_EXTENSIONS = ('.mp3', '.m4a', '.aac', '.flac', '.wav', '.ogg')

# ── Helpers ───────────────────────────────────────────────────

def get_lyrics(artist: str, title: str) -> Optional[str]:
    try:
        r = requests.get(
            f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(title)}",
            timeout=5,
        )
        lyrics = r.json().get("lyrics")
        if lyrics:
            return lyrics
    except Exception:
        pass
    try:
        song = genius.search_song(title, artist)
        return song.lyrics if song else None
    except Exception:
        return None


def split_lyrics(txt: str, num_parts: int = 8) -> list:
    lines = [l for l in txt.strip().split("\n") if l.strip()]
    size  = max(1, len(lines) // num_parts)
    parts = []
    for i in range(num_parts):
        chunk = lines[i * size : None if i == num_parts - 1 else (i + 1) * size]
        if chunk:
            parts.append("\n".join(chunk))
    return parts


def score_lyrics(txt: str):
    parts = split_lyrics(txt)
    weighted_sum, total_conf = 0.0, 0.0
    segments = []
    for part in parts:
        result = hartmann(part[:512])[0]
        top    = max(result, key=lambda x: x["score"])
        # collect all emotions with scores for richer frontend display
        all_emotions = {r["label"]: round(r["score"], 4) for r in result}
        weighted_sum += EMOTION_TO_SCALE[top["label"]] * top["score"]
        total_conf   += top["score"]
        segments.append({
            "label": top["label"],
            "score": round(top["score"], 4),
            "text": part,                  # actual lyrics chunk
            "emotions": all_emotions,      # all emotion scores for this chunk
        })
    avg = weighted_sum / total_conf if total_conf else 3.0
    if avg >= 4.5:   label = "pos"
    elif avg >= 3.5: label = "pos neu"
    elif avg >= 2.5: label = "neu"
    elif avg >= 1.5: label = "neu neg"
    else:            label = "neg"
    return round(avg, 4), label, segments


def fingerprint_song(filepath: str):
    try:
        results = acoustid.match(ACOUSTID_KEY, filepath)
        for score, _rid, title, artist in results:
            if score > 0.3:
                return artist, title
    except Exception:
        pass
    return None, None


def clean_tag(value: str) -> str:
    value = re.sub(r"\|.*", "", value)
    value = re.sub(r"https?\S+", "", value)
    value = re.sub(r"www\.\S+", "", value)
    value = re.sub(r"Via:.*", "", value, flags=re.IGNORECASE)
    return value.strip()


def clean_filename(filename: str):
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"_mp3_\d+$", "", name)
    name = re.sub(r"\[.*?\]", "", name)
    name = name.replace("_", " ").strip()
    if " - " in name:
        artist, title = name.split(" - ", 1)
        return artist.strip(), title.strip()
    return None, name.strip()


def resolve_metadata(filepath: str, filename: str):
    artist, title = fingerprint_song(filepath)
    if not artist or not title:
        try:
            audio = MutagenFile(filepath, easy=True)
            if audio:
                raw_artist = audio.get("artist", [None])[0]
                raw_title  = audio.get("title",  [None])[0]
                artist = artist or (clean_tag(raw_artist) if raw_artist else None)
                title  = title  or (clean_tag(raw_title)  if raw_title  else None)
        except Exception:
            pass
    if not artist or not title:
        fn_artist, fn_title = clean_filename(filename)
        artist = artist or fn_artist or "Unknown"
        title  = title  or fn_title  or os.path.splitext(filename)[0]
    return artist, title


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Pydantic models ───────────────────────────────────────────

class ManualRequest(BaseModel):
    artist: Optional[str] = ""
    title:  Optional[str] = ""
    lyrics: Optional[str] = ""


class AnalysisResult(BaseModel):
    artist:   str
    title:    str
    score:    float
    label:    str
    segments: list


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "LYRIMOD"}


@app.get("/scan/resolve")
def scan_resolve(name: str = Query(...)):
    import platform
    home = os.path.expanduser("~")
    system = platform.system()
    candidates = []

    music_bases = []
    if system == "Windows":
        music_bases = [
            os.path.join(home, "Music"),
            os.path.join(home, "Downloads"),
            os.path.join(home, "Desktop"),
        ]
        for drive in ["D", "E", "F"]:
            music_bases.append(f"{drive}:\\Music")
    else:
        music_bases = [
            os.path.join(home, "Music"),
            os.path.join(home, "Downloads"),
            os.path.join(home, "Desktop"),
            "/sdcard/Music",
            "/storage/emulated/0/Music",
        ]

    for base in music_bases:
        candidates.append(os.path.join(base, name))
        if os.path.isdir(base):
            try:
                for sub in os.listdir(base):
                    sub_path = os.path.join(base, sub)
                    if os.path.isdir(sub_path):
                        candidates.append(os.path.join(sub_path, name))
            except PermissionError:
                pass

    for path in candidates:
        if os.path.isdir(path):
            return {"path": path, "found": True}

    fallback = os.path.join(home, "Music", name)
    return {"path": fallback, "found": False}


@app.get("/scan/list")
def scan_list(folder: str = Query(...)):
    if not os.path.isdir(folder):
        raise HTTPException(400, f"Folder not found: {folder}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXTENSIONS)]
    return {"total": len(files), "files": files}


@app.get("/scan/stream")
async def scan_stream(folder: str = Query(...)):
    """
    SSE stream. Events:
      start   {total}
      result  AnalysisResult + {index, status:'ok'}
      skip    {index, filename, reason, status:'skip'}
      done    {processed, skipped, total}
    """
    if not os.path.isdir(folder):
        raise HTTPException(400, f"Folder not found: {folder}")

    audio_files = [f for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXTENSIONS)]
    total = len(audio_files)

    async def generate():
        yield sse("start", {"total": total})
        processed = 0
        skipped   = 0

        for idx, filename in enumerate(audio_files):
            filepath = os.path.join(folder, filename)
            await asyncio.sleep(0)

            suffix = os.path.splitext(filename)[-1].lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                with open(filepath, "rb") as src:
                    tmp.write(src.read())
                tmp_path = tmp.name

            try:
                artist, title = resolve_metadata(tmp_path, filename)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            if artist == "Unknown":
                skipped += 1
                yield sse("skip", {
                    "index": idx, "filename": filename,
                    "reason": "Could not identify song", "status": "skip",
                })
                continue

            lyrics = get_lyrics(artist, title)
            if not lyrics:
                skipped += 1
                yield sse("skip", {
                    "index": idx, "filename": filename,
                    "artist": artist, "title": title,
                    "reason": "Lyrics not found", "status": "skip",
                })
                continue

            try:
                avg, label, segments = score_lyrics(lyrics)
            except Exception as e:
                skipped += 1
                yield sse("skip", {
                    "index": idx, "filename": filename,
                    "artist": artist, "title": title,
                    "reason": f"Scoring error: {e}", "status": "skip",
                })
                continue

            processed += 1
            yield sse("result", {
                "index": idx, "status": "ok",
                "artist": artist, "title": title,
                "score": avg, "label": label, "segments": segments,
                "filepath": filepath,
            })

        yield sse("done", {"processed": processed, "skipped": skipped, "total": total})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/analyze/file", response_model=AnalysisResult)
async def analyze_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1].lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        artist, title = resolve_metadata(tmp_path, file.filename)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    lyrics = get_lyrics(artist, title)
    if not lyrics:
        raise HTTPException(404, f"Lyrics not found for '{title}' by {artist}")

    avg, label, segments = score_lyrics(lyrics)
    return AnalysisResult(artist=artist, title=title, score=avg, label=label, segments=segments)


@app.post("/analyze/manual", response_model=AnalysisResult)
async def analyze_manual(body: ManualRequest):
    artist = body.artist.strip() if body.artist else ""
    title  = body.title.strip()  if body.title  else ""
    lyrics = body.lyrics.strip() if body.lyrics else ""

    if not lyrics:
        if not artist and not title:
            raise HTTPException(400, "Provide artist/title or lyrics.")
        lyrics = get_lyrics(artist or "Unknown", title or "Unknown")
        if not lyrics:
            raise HTTPException(404, f"Lyrics not found for '{title}' by {artist}")

    avg, label, segments = score_lyrics(lyrics)
    return AnalysisResult(
        artist=artist or "Unknown",
        title=title or "Manual Entry",
        score=avg, label=label, segments=segments,
    )


class DeleteRequest(BaseModel):
    filepath: str


@app.post("/file/delete")
def file_delete(body: DeleteRequest):
    """Permanently delete a file from disk."""
    path = body.filepath
    if not path or not os.path.isfile(path):
        raise HTTPException(404, f"File not found: {path}")
    # Safety: only allow audio files
    if not path.lower().endswith(('.mp3', '.m4a', '.aac', '.flac', '.wav', '.ogg')):
        raise HTTPException(400, "Only audio files can be deleted.")
    try:
        os.remove(path)
        return {"deleted": True, "path": path}
    except PermissionError:
        raise HTTPException(403, "Permission denied — file may be in use.")
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")


@app.post("/file/delete-many")
def file_delete_many(body: dict):
    """Delete multiple files at once (for category-wide delete)."""
    paths = body.get("filepaths", [])
    results = {"deleted": [], "failed": []}
    for path in paths:
        if not path or not os.path.isfile(path):
            results["failed"].append({"path": path, "reason": "Not found"})
            continue
        if not path.lower().endswith(('.mp3', '.m4a', '.aac', '.flac', '.wav', '.ogg')):
            results["failed"].append({"path": path, "reason": "Not an audio file"})
            continue
        try:
            os.remove(path)
            results["deleted"].append(path)
        except Exception as e:
            results["failed"].append({"path": path, "reason": str(e)})
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
