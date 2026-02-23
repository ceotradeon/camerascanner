"""
Card Scanner API — Railway
POST /scan: Upload photo → detect & identify Pokemon cards → return JSON
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import imageio.v3 as iio
import cv2
import io
import os
import time
import asyncio
import threading
import logging

from scanner import (
    OCREngines,
    detect_all_blobs_from_array,
    orient_card,
    apply_orientation,
    process_single_blob_api,
    load_pokemon_db_from_json,
)

logger = logging.getLogger(__name__)

# ── Global state ──
engines = None
pokemon_db = {}
models_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines, pokemon_db, models_ready

    logger.info("Preloading OCR engines...")
    engines = OCREngines()
    _ = engines.easyocr  # Force eager load (~5-10s)
    logger.info("EasyOCR loaded.")

    db_path = os.path.join(os.path.dirname(__file__), "pokemon_names.json")
    pokemon_db = load_pokemon_db_from_json(db_path)
    logger.info(f"Pokemon DB: {len(pokemon_db)} entries")

    models_ready = True
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Card Scanner API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tradeon.jp",
        "https://www.tradeon.jp",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/health")
async def health():
    return {"status": "ok" if models_ready else "loading", "models_ready": models_ready}


@app.post("/scan")
async def scan_cards(image: UploadFile = File(...), max_cards: int = 9):
    if not models_ready:
        raise HTTPException(503, "Models still loading, try again in ~30s")

    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 10MB)")

    try:
        color_img = iio.imread(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    start = time.time()
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, _run_pipeline, color_img, max_cards
    )
    elapsed_ms = int((time.time() - start) * 1000)

    cards = []
    for r in results:
        bm = r.get("best_match")
        cards.append({
            "blob_index": r.get("blob_index", 0),
            "is_card": r.get("is_card", False),
            "flipped": r.get("flipped", False),
            "language": r.get("language", "unknown"),
            "ocr_engine": r.get("ocr_engine", "unknown"),
            "ocr_name": (r.get("ocr_detections") or {}).get("name"),
            "ocr_hp": (r.get("ocr_detections") or {}).get("hp_number"),
            "ocr_card_number": (r.get("ocr_detections") or {}).get("card_number"),
            "rarity_code": (r.get("card_id") or {}).get("rarity_code"),
            "rarity_name": (r.get("card_id") or {}).get("rarity_name"),
            "pokemon_match": _clean_pokemon_match(r.get("pokemon_match")),
            "best_match": {
                "name": bm["name"],
                "card_number": bm["card_number"],
                "set_name": bm.get("set", ""),
                "rarity": bm.get("rarity", ""),
                "image_url": bm.get("image_url", ""),
                "combined_score": bm.get("combined_score", 0),
                "blob_ratio": bm.get("blob_ratio", 0),
                "region_ratio": bm.get("region_ratio", 0),
                "db_row": bm.get("db_row", {}),
            } if bm else None,
            "candidates_count": r.get("candidates", 0),
            "summary": r.get("summary", ""),
        })

    return {
        "success": len(cards) > 0,
        "cards": cards,
        "total_cards": len(cards),
        "processing_time_ms": elapsed_ms,
    }


def _clean_pokemon_match(pm):
    """Strip non-serializable fields from pokemon_match."""
    if not pm:
        return None
    return {
        "matched_name": pm.get("matched_name", ""),
        "en": pm.get("en", ""),
        "dex": pm.get("dex", 0),
        "method": pm.get("method", ""),
        "dist": pm.get("dist", 0),
    }


def _run_pipeline(color_img, max_cards):
    """Full scanner pipeline (runs in thread pool)."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    blobs, raw_img, color_img, enhanced = detect_all_blobs_from_array(color_img)
    if not blobs:
        return []

    card_blobs = [b for b in blobs if b["is_card_like"]]
    if not card_blobs:
        card_blobs = blobs[:5]
    card_blobs = card_blobs[:max_cards]

    # Phase 1: Orient all cards
    orient_data_list = []
    for i, blob in enumerate(card_blobs):
        od = orient_card(blob, color_img, enhanced, engines, i)
        orient_data_list.append(od)

    # Phase 2: Majority vote
    WEAK_THRESHOLD = 3
    total_signal = sum(od["orient_score"] for od in orient_data_list)
    majority_flip = total_signal < 0

    orientations = []
    for i, od in enumerate(orient_data_list):
        score = od["orient_score"]
        if score >= WEAK_THRESHOLD:
            flip = False
        elif score <= -WEAK_THRESHOLD:
            flip = True
        else:
            flip = score < 0
        if len(card_blobs) >= 3 and abs(score) < WEAK_THRESHOLD:
            flip = majority_flip
        orientations.append(flip)

    # Phase 3: Process each card
    yomitoku_lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process(args):
        i, blob = args
        return process_single_blob_api(
            blob, color_img, enhanced, clahe,
            blob_index=i, engines=engines, pokemon_db=pokemon_db,
            orient_data=orient_data_list[i], card_flipped=orientations[i],
            yomitoku_lock=yomitoku_lock,
        )

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_process, (i, blob)): i
                   for i, blob in enumerate(card_blobs)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_idx[idx] = future.result()
            except Exception as e:
                results_by_idx[idx] = {"summary": f"Error: {e}"}

    return [results_by_idx[i] for i in range(len(card_blobs))]
