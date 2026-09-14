"""main.py

FastAPI server for Google Fonts Visual Matcher with Dynamic On-The-Fly Rendering.
1. Preprocesses user image with contrast normalization (preserving anti-aliased stroke weights).
2. Extracts text from user uploads using Tesseract OCR (with optional user override).
3. Dynamically renders that exact text across all candidate fonts in the library using Pillow at high resolution.
4. Batch extracts 512-d visual embeddings using OpenAI CLIP to compare pure typographic shapes.
5. Computes cosine similarities directly, groups by family to select the best-matching weight,
   and returns the top closest Google Font candidates.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import urllib.parse
import urllib.request
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Prevent duplicate libomp crash on macOS when torch and faiss are loaded together
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import pytesseract
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FONTS_DIR = Path("fonts")
METADATA_PATH = Path("metadata.json")
CATALOG_PATH = Path("google_fonts_catalog.json")
DB_PATH = Path("font_cache.db")
CANVAS_SIZE = (224, 224)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
DINO_MODEL_NAME = "dinov2_vits14_reg"
FINE_EMBEDDING_DIM = 384  # DINOv2-S hidden size
FINE_STAGE_LIMIT = 60  # candidates refined with per-glyph embeddings

# Typographic macro archetypes for Zero-Shot CLIP coarse classification
TYPOGRAPHY_ARCHETYPES: list[tuple[str, str]] = [
    ("sans-serif", "sans-serif clean neutral modern grotesque geometric text typography font"),
    ("serif", "serif traditional classic editorial book bracketed slab typography font"),
    ("monospace", "monospace programming typewriter fixed-width code typography font"),
    ("handwriting", "handwriting cursive script handwritten calligraphy typography font"),
    ("display", "decorative novelty artistic ornate stylized graffiti display typography font"),
]

# Universal workhorse core fonts: guaranteed in candidate pool across all queries
UNIVERSAL_CORE_FONTS: list[str] = [
    "Inter", "Roboto", "Open Sans", "Montserrat", "Lato",
    "Poppins", "Raleway", "Nunito", "Work Sans", "DM Sans",
    "Merriweather", "Playfair Display", "Lora", "PT Serif",
    "Oswald", "JetBrains Mono", "Space Grotesk", "Sora", "Albert Sans"
]

# In-memory LRU font micro-subset cache: (family, weight, text) -> bytes
_FONT_CACHE: OrderedDict[tuple[str, str, str], bytes] = OrderedDict()
MAX_FONT_CACHE_SIZE = 1000

# Curated complementary font pairing directory
FONT_PAIRINGS: dict[str, list[str]] = {
    "inter": ["Merriweather", "Playfair Display", "Lora", "Libre Baskerville"],
    "roboto": ["Roboto Slab", "Playfair Display", "Lora", "Merriweather"],
    "open sans": ["Merriweather", "Lora", "Roboto Slab", "Bitter"],
    "montserrat": ["Lora", "Merriweather", "Cormorant Garamond", "EB Garamond"],
    "lato": ["Playfair Display", "Merriweather", "Cinzel", "PT Serif"],
    "poppins": ["Lora", "Playfair Display", "Merriweather", "Bodoni Moda"],
    "raleway": ["Merriweather", "Lora", "Playfair Display", "EB Garamond"],
    "work sans": ["Playfair Display", "Merriweather", "Bitter", "PT Serif"],
    "dm sans": ["DM Serif Display", "Playfair Display", "Lora", "Prata"],
    "outfit": ["Playfair Display", "Lora", "Castoro", "Bodoni Moda"],
    "plus jakarta sans": ["Lora", "Merriweather", "Playfair Display"],
    "merriweather": ["Open Sans", "Lato", "Inter", "Roboto"],
    "playfair display": ["Inter", "Montserrat", "Lato", "Poppins"],
    "lora": ["Montserrat", "Lato", "Inter", "Open Sans"],
    "pt serif": ["PT Sans", "Inter", "Open Sans", "Roboto"],
    "oswald": ["Quicksand", "Open Sans", "Lato", "Merriweather"],
    "bebas neue": ["Montserrat", "Open Sans", "Roboto", "Lato"],
    "jetbrains mono": ["Inter", "Plus Jakarta Sans", "Work Sans"],
    "space mono": ["Space Grotesk", "Inter", "Work Sans"],
    "space grotesk": ["Space Mono", "Inter", "Lora"],
    "pacifico": ["Open Sans", "Montserrat", "Quicksand"],
    "caveat": ["Open Sans", "Montserrat", "Lato"],
    "albert sans": ["Merriweather", "Lora", "Playfair Display"],
    "sora": ["Lora", "Merriweather", "Playfair Display"],
}

def get_font_pairings(family: str, category: str) -> list[str]:
    """Return 2-3 curated complementary Google Fonts for typography pairing."""
    fam_lower = family.lower().strip()
    if fam_lower in FONT_PAIRINGS:
        return FONT_PAIRINGS[fam_lower][:3]
    if category == "serif":
        return ["Inter", "Open Sans", "Montserrat"]
    elif category in ("sans-serif", "monospace"):
        return ["Merriweather", "Playfair Display", "Lora"]
    elif category == "display":
        return ["Inter", "Roboto", "Open Sans"]
    else:
        return ["Montserrat", "Open Sans", "Lato"]

# ---------------------------------------------------------------------------
# Persistent SQLite Micro-Font Cache
# ---------------------------------------------------------------------------

def init_cache_db():
    """Initialize persistent SQLite table for downloaded font micro-subsets."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS font_subsets (
                    family TEXT NOT NULL,
                    weight TEXT NOT NULL,
                    text TEXT NOT NULL,
                    font_bytes BLOB NOT NULL,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (family, weight, text)
                )
            """)
            conn.commit()
    except Exception as exc:
        logger.warning(f"Failed to initialize font cache database: {exc}")


def db_get_font(family: str, weight: str, text: str) -> bytes | None:
    """Read micro-subset font bytes from SQLite persistent cache."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT font_bytes FROM font_subsets WHERE family=? AND weight=? AND text=?",
                (family.lower(), str(weight), text),
            )
            row = cursor.fetchone()
            if row:
                return row[0]
    except Exception as exc:
        logger.debug(f"SQLite cache read error: {exc}")
    return None


def db_put_font(family: str, weight: str, text: str, data: bytes) -> None:
    """Write micro-subset font bytes to SQLite persistent cache."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO font_subsets (family, weight, text, font_bytes) VALUES (?, ?, ?, ?)",
                (family.lower(), str(weight), text, data),
            )
            conn.commit()
    except Exception as exc:
        logger.debug(f"SQLite cache write error: {exc}")


# Configure Tesseract binary path
TESSERACT_CMD = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
if Path(TESSERACT_CMD).exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD)
    logger.info(f"Configured Tesseract binary: {TESSERACT_CMD}")
else:
    logger.warning("Tesseract binary not found at default locations. OCR may rely on system PATH.")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class FontMatch(BaseModel):
    """Schema for a matched Google Font candidate."""
    family: str = Field(..., description="Font family name")
    variant: str = Field(..., description="Font variant or weight style")
    category: str = Field("sans-serif", description="Font typographic category")
    similarity_score: float = Field(..., description="Calibrated similarity score [0.0, 1.0]")
    raw_score: float = Field(..., description="Raw CLIP cosine similarity score [-1.0, 1.0]")
    google_fonts_url: str = Field(..., description="Direct URL to font on Google Fonts")
    recommended_pairings: list[str] = Field(default_factory=list, description="Complementary Google Font pairing suggestions")


class MatchResponse(BaseModel):
    """Schema for font matching response."""
    detected_text: str = Field(..., description="The text string used for comparison")
    detected_category: str = Field("sans-serif", description="Detected typographic category")
    category_confidence: float = Field(1.0, description="Confidence score for detected category")
    category_probabilities: dict[str, float] = Field(default_factory=dict, description="Class probabilities across categories")
    estimated_weight: int = Field(400, description="Estimated optical font weight (100-900)")
    detected_skew: float = Field(0.0, description="Detected rotation skew angle in degrees")
    matches: list[FontMatch] = Field(..., description="Top ranked font matches")
    source: str = Field("cloud_subset", description="Candidates source: cloud_subset or local_fallback")


class HealthStatus(BaseModel):
    """Server health status schema."""
    status: str
    device: str
    model: str
    embedding_dimension: int
    fine_model: str
    fine_embedding_dimension: int
    catalog_fonts: int
    indexed_fonts: int
    index_loaded: bool


# ---------------------------------------------------------------------------
# Device & Model Loading
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select compute device: CUDA, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_model(
    model_name: str = CLIP_MODEL_NAME,
    device: torch.device = torch.device("cpu"),
) -> tuple[CLIPModel, CLIPProcessor]:
    """Load pre-trained OpenAI CLIP model and processor from Hugging Face."""
    logger.info(f"Loading CLIP model and processor ('{model_name}')...")
    try:
        processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        model = CLIPModel.from_pretrained(model_name, local_files_only=True)
    except Exception:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, processor


def load_dino_model(
    model_name: str = DINO_MODEL_NAME,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Any]:
    """Load DINOv2 via torch.hub (Meta's official repo, public weights) and return (model, transform).

    Used for the fine-grained per-glyph matching stage. HuggingFace mirrors of
    facebook/dinov2* are gated; torch.hub pulls weights from Meta's public CDN instead.
    """
    from torchvision.transforms import v2 as T

    logger.info(f"Loading DINOv2 fine-matcher '{model_name}' via torch.hub...")
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.eval()
    model.to(device)

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def embed_batch_dino(
    model: torch.nn.Module,
    transform: Any,
    images: list[Image.Image],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed images with DINOv2 and return L2-normalized CLS-token vectors (N, D)."""
    if not images:
        return np.empty((0, FINE_EMBEDDING_DIM), dtype=np.float32)
    features: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        batch = torch.stack([transform(img) for img in chunk]).to(device)
        with torch.no_grad():
            outputs = model.forward_features(batch)
        if isinstance(outputs, dict):
            cls_tokens = outputs["x_norm_clstoken"]
        elif outputs.ndim == 3:
            cls_tokens = outputs[:, 0, :]
        else:
            cls_tokens = outputs
        cls_tokens = cls_tokens / cls_tokens.norm(p=2, dim=-1, keepdim=True)
        features.append(cls_tokens.cpu().numpy())
    return np.vstack(features).astype(np.float32)


def load_font_library() -> list[dict[str, Any]]:
    """Scan FONTS_DIR and pair each .ttf font with its metadata."""
    metadata_map: dict[str, dict[str, Any]] = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
                for item in raw_meta:
                    fname = item.get("file_name")
                    if fname:
                        metadata_map[fname] = item
        except Exception as exc:
            logger.error(f"Failed to read metadata.json: {exc}")

    library: list[dict[str, Any]] = []
    for font_file in sorted(FONTS_DIR.glob("*.ttf")):
        meta = metadata_map.get(font_file.name)
        if not meta:
            # Fallback metadata constructed via TrueType table or filename
            try:
                tt_font = ImageFont.truetype(str(font_file), 24)
                family, variant = tt_font.getname()
                if "-Bold" in font_file.name:
                    variant = "Bold"
                elif "-Regular" in font_file.name:
                    variant = "Regular"
            except Exception:
                name_stem = font_file.stem
                parts = name_stem.split("-")
                family = parts[0].replace("_", " ")
                variant = parts[1] if len(parts) > 1 else "Regular"
            meta = {
                "family": family,
                "variant": variant,
                "file_name": font_file.name,
                "google_fonts_url": f"https://fonts.google.com/specimen/{family.replace(' ', '+')}",
            }
        library.append({
            "font_path": str(font_file),
            "file_name": font_file.name,
            "family": meta.get("family", font_file.stem),
            "variant": meta.get("variant", "Regular"),
            "google_fonts_url": meta.get(
                "google_fonts_url",
                f"https://fonts.google.com/specimen/{meta.get('family', '').replace(' ', '+')}",
            ),
        })

    logger.info(f"Loaded font library with {len(library)} fonts from '{FONTS_DIR.resolve()}'.")
    return library


# ---------------------------------------------------------------------------
# Image Preprocessing & OCR (Stroke Weight Preserving)
# ---------------------------------------------------------------------------

def normalize_text_crop(
    img: Image.Image,
    target_size: tuple[int, int] = CANVAS_SIZE,
    padding_ratio: float = 0.1,
    apply_sharpen: bool = True,
) -> Image.Image:
    """Standardized normalization pipeline for font text crops with intelligent upscaling.
    
    1. Locates tight bounding box around text ink.
    2. Crops strictly to bounding box.
    3. Intelligent Upscaling for Small Crops:
       If width or height is below 150px, upscales cleanly using LANCZOS and applies
       PIL's ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2) to recover
       crisp character boundaries and prevent fuzzy gray gradients when scaled to canvas.
    4. Centers cropped text on a square white canvas to preserve natural aspect ratio.
    5. Resizes to target_size using LANCZOS resampling.
    """
    gray = img.convert("L")
    arr = np.array(gray)

    # Invert so ink pixels are positive
    if np.mean(arr) > 128.0:
        ink_mask = ImageOps.invert(gray)
        crop_source = gray.convert("RGB")
    else:
        ink_mask = gray
        crop_source = ImageOps.invert(gray).convert("RGB")

    thresh_mask = ink_mask.point(lambda p: 255 if p > 25 else 0)
    bbox = thresh_mask.getbbox() or ink_mask.getbbox() or (0, 0, gray.width, gray.height)

    cropped = crop_source.crop(bbox)
    cw, ch = cropped.size

    # 1. Intelligent Upscaling for Small Crops (< 150px on either dimension)
    if cw < 150 or ch < 150:
        scale = min(max(150.0 / max(cw, 1), 150.0 / max(ch, 1)), 4.0)
        scale = max(scale, 1.5)
        new_w = int(round(cw * scale))
        new_h = int(round(ch * scale))
        cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if apply_sharpen:
            cropped = cropped.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))
        cw, ch = cropped.size

    # 2. Square canvas placement with aspect ratio preservation
    max_side = max(cw, ch)
    padding = max(int(max_side * padding_ratio), 1)
    canvas_dim = max_side + (2 * padding)

    canvas = Image.new("RGB", (canvas_dim, canvas_dim), color=(255, 255, 255))
    offset_x = (canvas_dim - cw) // 2
    offset_y = (canvas_dim - ch) // 2
    canvas.paste(cropped, (offset_x, offset_y))

    # 3. Resize final square canvas to target_size using LANCZOS
    return canvas.resize(target_size, Image.Resampling.LANCZOS)


def deskew_text_image(img: Image.Image, max_angle: float = 12.0, step: float = 0.5) -> tuple[Image.Image, float]:
    """Detect and correct rotation tilt (skew) in text image using projection profile variance."""
    gray = img.convert("L")
    arr = np.array(gray)
    if np.mean(arr) > 128:
        ink = (arr < 200).astype(np.float32)
    else:
        ink = (arr > 55).astype(np.float32)

    if ink.sum() < 50:
        return img, 0.0

    best_angle = 0.0
    max_var = -1.0
    angles = np.arange(-max_angle, max_angle + step, step)

    for ang in angles:
        rot = Image.fromarray(ink).rotate(float(ang), resample=Image.NEAREST, fillcolor=0)
        prof = np.sum(np.array(rot), axis=1)
        var = float(np.var(prof))
        if var > max_var:
            max_var = var
            best_angle = float(ang)

    if abs(best_angle) >= 0.7:
        corrected = img.rotate(best_angle, resample=Image.BICUBIC, fillcolor=255)
        logger.info(f"Deskewing applied: {-best_angle:.1f}° rotation corrected.")
        return corrected, round(float(-best_angle), 1)

    return img, 0.0


def estimate_stroke_weight(img: Image.Image) -> tuple[int, float]:
    """Estimate optical font weight (300, 400, 600, 700, 800) based on stroke-to-height ratio."""
    gray = img.convert("L")
    arr = np.array(gray)
    if np.mean(arr) > 128:
        ink = (arr < 200).astype(np.uint8)
    else:
        ink = (arr > 55).astype(np.uint8)

    y_indices, x_indices = np.where(ink)
    if len(y_indices) == 0:
        return 400, 0.10

    h = y_indices.max() - y_indices.min() + 1
    horiz_runs = []
    for y in range(y_indices.min(), y_indices.max() + 1, 2):
        row = ink[y, :]
        cur = 0
        for val in row:
            if val == 1:
                cur += 1
            elif cur > 0:
                if cur < h * 0.45:
                    horiz_runs.append(cur)
                cur = 0
        if 0 < cur < h * 0.45:
            horiz_runs.append(cur)

    if not horiz_runs:
        return 400, 0.10

    med_stroke = float(np.median(horiz_runs))
    ratio = med_stroke / max(h, 1)

    if ratio < 0.08:
        weight = 300
    elif ratio < 0.125:
        weight = 400
    elif ratio < 0.165:
        weight = 600
    elif ratio < 0.22:
        weight = 700
    else:
        weight = 800

    return weight, round(float(ratio), 4)


def extract_multiscale_crops(img: Image.Image, text: str) -> tuple[Image.Image, Image.Image | None]:
    """Extract full-word normalized crop and a focused lead-glyph patch for high-resolution details."""
    full_crop = normalize_text_crop(img, target_size=CANVAS_SIZE, padding_ratio=0.1, apply_sharpen=True)
    if len(text) > 7 or " " in text:
        gray = img.convert("L")
        arr = np.array(gray)
        ink = (arr < 200) if np.mean(arr) > 128 else (arr > 55)
        y_indices, x_indices = np.where(ink)
        if len(x_indices) > 0 and len(y_indices) > 0:
            min_x, max_x = x_indices.min(), x_indices.max()
            min_y, max_y = y_indices.min(), y_indices.max()
            patch_w = max(int((max_x - min_x) * 0.35), 20)
            patch_box = (min_x, min_y, min(min_x + patch_w, max_x), max_y)
            patch = img.crop(patch_box)
            patch_crop = normalize_text_crop(patch, target_size=CANVAS_SIZE, padding_ratio=0.1, apply_sharpen=True)
            return full_crop, patch_crop

    return full_crop, None


def preprocess_user_image(img: Image.Image) -> tuple[Image.Image, Image.Image, int, float]:
    """Preprocess user image with automatic deskewing, polarity correction, and stroke estimation.
    Returns (query_crop, cleaned_full, estimated_weight, detected_skew).
    """
    # 1. Multi-frame images
    if hasattr(img, "seek"):
        try:
            img.seek(0)
        except Exception:
            pass

    # 2. Transparency handling: composite over solid white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba_img = img.convert("RGBA")
        white_canvas = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_canvas, rgba_img).convert("RGB")

    # 3. Grayscale conversion
    gray_img = img.convert("L")
    arr = np.array(gray_img)

    # 4. Background polarity correction (ensure dark text on light background)
    h, w = arr.shape
    if h > 1 and w > 1:
        border_pixels = np.concatenate([
            arr[0, :],
            arr[h - 1, :],
            arr[:, 0],
            arr[:, w - 1],
        ])
        background_val = float(np.median(border_pixels))
    else:
        background_val = float(np.median(arr))

    if background_val < 128.0:
        gray_img = ImageOps.invert(gray_img)

    # 5. Controlled Contrast Stretching:
    enhanced = ImageOps.autocontrast(gray_img, cutoff=2)
    cleaned_full = enhanced.point(lambda p: 255 if p >= 245 else int(p * 255 / 245))

    # 6. Automatic Projection Deskewing:
    deskewed_full, detected_skew = deskew_text_image(cleaned_full)

    # 7. Optical Stroke-to-Height Weight Estimation:
    estimated_weight, stroke_ratio = estimate_stroke_weight(deskewed_full)

    # 8. Standardized crop for CLIP query vector
    query_crop = normalize_text_crop(deskewed_full.convert("RGB"), target_size=CANVAS_SIZE, padding_ratio=0.1, apply_sharpen=True)

    return query_crop, deskewed_full, estimated_weight, detected_skew


def extract_text_from_image(cleaned_img: Image.Image) -> str:
    """Extract text from contrast-normalized image using Tesseract OCR."""
    try:
        # Try single text line (PSM 7) first for crisp words
        text = pytesseract.image_to_string(cleaned_img, config="--psm 7").strip()
        if not text:
            # Fallback to standard uniform text block (PSM 6)
            text = pytesseract.image_to_string(cleaned_img, config="--psm 6").strip()
        if not text:
            # Fallback to default PSM
            text = pytesseract.image_to_string(cleaned_img).strip()

        cleaned = " ".join(text.split())
        return cleaned
    except Exception as exc:
        logger.warning(f"OCR extraction encountered error: {exc}")
        return ""


def render_font_sample(font_path: str, text: str, font_size: int = 72) -> Image.Image:
    """Dynamically render candidate font text on clean white canvas with matching edge sharpness and padding."""
    try:
        font = ImageFont.truetype(font_path, font_size)
        fname = Path(font_path).name
        if "-Bold" in fname or "Bold" in fname:
            try:
                font.set_variation_by_name("Bold")
            except Exception:
                pass
    except Exception:
        font = ImageFont.load_default()

    canvas_w = max(600, len(text) * 60)
    canvas_h = 240
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 50), text, fill=(0, 0, 0), font=font)

    # Pass through normalize_text_crop to ensure identical sharpness profile and padding ratio
    return normalize_text_crop(canvas, target_size=CANVAS_SIZE, padding_ratio=0.1, apply_sharpen=True)


def render_font_sample_from_bytes(font_bytes: bytes, text: str, font_size: int = 72) -> Image.Image:
    """Dynamically render candidate font text from in-memory TTF bytes without saving to disk."""
    try:
        font = ImageFont.truetype(io.BytesIO(font_bytes), font_size)
    except Exception:
        font = ImageFont.load_default()

    canvas_w = max(600, len(text) * 60)
    canvas_h = 240
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 50), text, fill=(0, 0, 0), font=font)

    return normalize_text_crop(canvas, target_size=CANVAS_SIZE, padding_ratio=0.1, apply_sharpen=True)


def render_candidate_parts(renderer, text: str, max_chars: int = 8) -> list[Image.Image]:
    """Render a candidate font as a whole-word crop plus individual per-glyph crops."""
    parts = [renderer(text)]
    count = 0
    for ch in text:
        if not ch.strip():
            continue
        if count >= max_chars:
            break
        parts.append(renderer(ch))
        count += 1
    return parts


def extract_query_glyph_crops(cleaned_img: Image.Image, max_glyphs: int = 12) -> list[Image.Image]:
    """Extract per-character ink crops from the user image via Tesseract char boxes."""
    crops: list[Image.Image] = []
    try:
        boxes = pytesseract.image_to_boxes(cleaned_img, config="--psm 7")
    except Exception as exc:
        logger.debug(f"Glyph box extraction failed: {exc}")
        return crops

    for line in boxes.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        glyph = parts[0]
        if not glyph.isprintable() or glyph == " ":
            continue
        try:
            left, bottom, right, top = (int(x) for x in parts[1:5])
        except ValueError:
            continue
        if right - left < 2 or bottom - top < 2:
            continue
        pad = 3
        box = (
            max(0, left - pad),
            max(0, top - pad),
            min(cleaned_img.width, right + pad),
            min(cleaned_img.height, bottom + pad),
        )
        crop = cleaned_img.crop(box)
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        crops.append(
            normalize_text_crop(crop, target_size=CANVAS_SIZE, padding_ratio=0.1, apply_sharpen=False)
        )
        if len(crops) >= max_glyphs:
            break
    return crops


def compose_fine_feature(
    normalized_parts: np.ndarray,
    full_weight: float = 0.75,
) -> np.ndarray:
    """Blend whole-word and per-glyph embeddings into a single L2-normalized vector."""
    parts = np.asarray(normalized_parts, dtype=np.float32)
    if parts.ndim != 2 or parts.shape[0] == 0:
        raise ValueError("compose_fine_feature requires a 2-D array of normalized embeddings")
    if parts.shape[0] == 1:
        return parts[0]
    full = parts[0]
    glyph_mean = parts[1:].mean(axis=0)
    glyph_mean = glyph_mean / (float(np.linalg.norm(glyph_mean)) or 1.0)
    combined = full_weight * full + (1.0 - full_weight) * glyph_mean
    return combined / (float(np.linalg.norm(combined)) or 1.0)


def embed_batch_dino(
    model: torch.nn.Module,
    transform: Any,
    images: list[Image.Image],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed images with DINOv2 and return L2-normalized CLS-token vectors (N, D)."""
    if not images:
        return np.empty((0, FINE_EMBEDDING_DIM), dtype=np.float32)
    features: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        batch = torch.stack([transform(img) for img in chunk]).to(device)
        with torch.no_grad():
            outputs = model.forward_features(batch)
        if isinstance(outputs, dict):
            cls_tokens = outputs["x_norm_clstoken"]
        elif outputs.ndim == 3:
            cls_tokens = outputs[:, 0, :]
        else:
            cls_tokens = outputs
        cls_tokens = cls_tokens / cls_tokens.norm(p=2, dim=-1, keepdim=True)
        features.append(cls_tokens.cpu().numpy())
    return np.vstack(features).astype(np.float32)


# Backward-compatible alias
render_font_candidate = render_font_sample


def calibrate_score(raw_score: float, baseline: float = 0.60, power: float = 1.8) -> float:
    """Dynamic and forgiving score calibration."""
    clamped = max(0.0, (raw_score - baseline) / (1.0 - baseline))
    return round(float(clamped ** power), 4)


# ---------------------------------------------------------------------------
# Two-Stage In-Memory Subsetting & Classification Pipeline
# ---------------------------------------------------------------------------

def cache_get_font(family: str, weight: str, text: str) -> bytes | None:
    """Retrieve font micro-subset bytes from in-memory LRU cache or persistent SQLite cache."""
    key = (family.lower(), str(weight), text)
    if key in _FONT_CACHE:
        _FONT_CACHE.move_to_end(key)
        return _FONT_CACHE[key]
    
    # Fallback to persistent SQLite disk cache
    db_data = db_get_font(family, weight, text)
    if db_data is not None:
        _FONT_CACHE[key] = db_data
        _FONT_CACHE.move_to_end(key)
        if len(_FONT_CACHE) > MAX_FONT_CACHE_SIZE:
            _FONT_CACHE.popitem(last=False)
        return db_data
    return None


def cache_put_font(family: str, weight: str, text: str, data: bytes) -> None:
    """Store font micro-subset bytes in in-memory LRU cache and persistent SQLite cache."""
    key = (family.lower(), str(weight), text)
    _FONT_CACHE[key] = data
    _FONT_CACHE.move_to_end(key)
    if len(_FONT_CACHE) > MAX_FONT_CACHE_SIZE:
        _FONT_CACHE.popitem(last=False)
    # Persist to SQLite DB
    db_put_font(family, weight, text, data)


def fetch_family_subsets(family: str, text: str, target_weight: int = 400) -> list[dict[str, Any]]:
    """Fetch on-demand TrueType micro-font subsets (~2-8 KB) for requested family directly into RAM.
    Supports optical target_weight along with default 400/700 anchors.
    Returns list of candidate dicts: family, weight, variant, bytes.
    """
    requested_weights = sorted(list({int(target_weight), 400, 700}))
    cached_results = []
    all_cached = True

    for w in requested_weights:
        c = cache_get_font(family, str(w), text)
        if c:
            v = "Regular"
            if w >= 700:
                v = "Bold"
            elif w >= 600:
                v = "SemiBold"
            elif w <= 300:
                v = "Light"
            cached_results.append({
                "family": family,
                "weight": str(w),
                "variant": v,
                "bytes": c,
            })
        else:
            all_cached = False

    if all_cached and cached_results:
        return cached_results

    enc_family = family.replace(" ", "+")
    enc_text = urllib.parse.quote(text)
    weights_param = ";".join(str(w) for w in requested_weights)
    css_url = f"https://fonts.googleapis.com/css2?family={enc_family}:wght@{weights_param}&text={enc_text}"
    req = urllib.request.Request(css_url, headers={"User-Agent": "Mozilla/5.0"})

    css_content = ""
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            css_content = resp.read().decode("utf-8")
    except Exception:
        # Fallback to standard 400;700 weights
        try:
            css_url_std = f"https://fonts.googleapis.com/css2?family={enc_family}:wght@400;700&text={enc_text}"
            req_std = urllib.request.Request(css_url_std, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_std, timeout=4) as resp:
                css_content = resp.read().decode("utf-8")
        except Exception:
            # Fallback to unconstrained weight
            css_url_any = f"https://fonts.googleapis.com/css2?family={enc_family}&text={enc_text}"
            req_any = urllib.request.Request(css_url_any, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req_any, timeout=4) as resp:
                    css_content = resp.read().decode("utf-8")
            except Exception:
                return cached_results

    blocks = re.findall(r"font-weight:\s*(\d+);.*?src:\s*url\((https://[^\)]+)\)", css_content, re.DOTALL)
    if not blocks:
        urls = re.findall(r"src:\s*url\((https://[^\)]+)\)", css_content)
        if urls:
            blocks = [("400", urls[0])]

    results = []
    seen_weights = set()
    for weight, url in blocks:
        if weight in seen_weights:
            continue
        seen_weights.add(weight)
        w_int = int(weight)
        variant = "Regular"
        if w_int >= 700:
            variant = "Bold"
        elif w_int >= 600:
            variant = "SemiBold"
        elif w_int <= 300:
            variant = "Light"

        cached = cache_get_font(family, weight, text)
        if cached:
            results.append({
                "family": family,
                "weight": weight,
                "variant": variant,
                "bytes": cached,
            })
            continue

        try:
            req_font = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_font, timeout=4) as f_resp:
                data = f_resp.read()
                cache_put_font(family, weight, text, data)
                results.append({
                    "family": family,
                    "weight": weight,
                    "variant": variant,
                    "bytes": data,
                })
        except Exception:
            pass

    if not results and cached_results:
        return cached_results

    return results


def load_google_fonts_catalog() -> list[dict[str, Any]]:
    """Load comprehensive Google Fonts catalog with 1,580+ fonts."""
    if CATALOG_PATH.exists():
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                catalog = json.load(f)
                logger.info(f"Loaded {len(catalog)} fonts from catalog '{CATALOG_PATH.resolve()}'.")
                return catalog
        except Exception as exc:
            logger.error(f"Failed to read {CATALOG_PATH}: {exc}")
    logger.warning("Catalog file not found. Falling back to local ./fonts.")
    return []


def classify_typography(
    crop_img: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    """Classify text crop into macro typographic categories via Zero-Shot CLIP."""
    labels = [text for _, text in TYPOGRAPHY_ARCHETYPES]
    keys = [key for key, _ in TYPOGRAPHY_ARCHETYPES]

    inputs = processor(text=labels, images=crop_img.convert("RGB"), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()

    prob_dict = {k: round(float(p), 4) for k, p in zip(keys, probs)}
    best_idx = int(np.argmax(probs))
    best_cat = keys[best_idx]
    best_conf = round(float(probs[best_idx]), 4)

    return best_cat, best_conf, prob_dict


# ---------------------------------------------------------------------------
# Application Lifespan & FastAPI App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load CLIP model and font library on startup."""
    init_cache_db()
    device = get_device()
    logger.info(f"Using compute device: {device}")
    app.state.device = device
    model, processor = load_clip_model(CLIP_MODEL_NAME, device)
    app.state.model = model
    app.state.processor = processor

    try:
        dino_model, dino_transform = load_dino_model(DINO_MODEL_NAME, device)
        app.state.dino_model = dino_model
        app.state.dino_transform = dino_transform
        logger.info("DINOv2 per-glyph fine-matcher loaded for Stage 2.")
    except Exception as exc:
        app.state.dino_model = None
        app.state.dino_transform = None
        logger.warning(f"DINOv2 fine-matcher failed to load ({exc}); falling back to CLIP-only matching.")

    catalog = load_google_fonts_catalog()
    app.state.catalog = catalog

    font_library = load_font_library()
    app.state.font_library = font_library

    logger.info(f"Catalog contains {len(catalog)} fonts; local fallback contains {len(font_library)} fonts.")
    print(f"✓ Google Fonts Catalog: {len(catalog)} fonts ready for on-demand cloud subsetting.")
    print(f"✓ Local Fallback Pool: {len(font_library)} fonts in ./fonts.")

    yield

    logger.info("Shutting down Google Fonts Visual Matcher server.")


app = FastAPI(
    title="Google Fonts Visual Matcher API",
    description="Dynamic on-the-fly visual typography matching using OCR and OpenAI CLIP embeddings.",
    version="3.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the local frontend interface."""
    index_file = Path("index.html")
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({
        "message": "Google Fonts Visual Matcher API is running. Place index.html in the project root to access the UI.",
        "docs": "/docs",
        "match_endpoint": "/api/match",
    })


@app.get("/index.html", include_in_schema=False)
async def serve_index_html():
    """Serve index.html at explicit /index.html route."""
    return await serve_index()


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Health check endpoint reporting device, catalog, and local fallback status."""
    catalog: list[dict[str, Any]] = getattr(app.state, "catalog", [])
    font_library: list[dict[str, Any]] = getattr(app.state, "font_library", [])
    device: torch.device = getattr(app.state, "device", torch.device("cpu"))

    return HealthStatus(
        status="healthy",
        device=str(device),
        model=CLIP_MODEL_NAME,
        embedding_dimension=EMBEDDING_DIM,
        fine_model=getattr(app.state, "dino_model", None) is not None and DINO_MODEL_NAME or "disabled_fallback_clip",
        fine_embedding_dimension=FINE_EMBEDDING_DIM,
        catalog_fonts=len(catalog),
        indexed_fonts=len(font_library),
        index_loaded=len(catalog) > 0 or len(font_library) > 0,
    )


@app.post("/api/reload", tags=["Management"])
async def reload_fonts():
    """Reload catalog and local font library without restarting the server."""
    catalog = load_google_fonts_catalog()
    font_library = load_font_library()
    app.state.catalog = catalog
    app.state.font_library = font_library
    return {
        "status": "reloaded",
        "catalog_fonts": len(catalog),
        "local_fallback_fonts": len(font_library),
    }


@app.post(
    "/api/match",
    response_model=MatchResponse,
    summary="Match an uploaded image to Google Fonts via two-stage filtering and on-demand in-memory subsetting",
    tags=["Inference"],
)
async def match_font(
    file: UploadFile = File(...),
    override_text: str | None = Form(None),
    category_override: str | None = Form(None),
    top_k: int = Form(3),
) -> MatchResponse:
    """Accepts an uploaded image of text, extracts the text string via OCR (or user override),
    classifies typographic style (Stage 1), streams candidate micro-font subsets into RAM (Stage 2),
    and matches them via batch CLIP embeddings.
    """
    catalog: list[dict[str, Any]] = getattr(app.state, "catalog", [])
    font_library: list[dict[str, Any]] = getattr(app.state, "font_library", [])
    if not catalog and not font_library:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Font catalog and local library are empty.",
        )

    # 1. Read and validate uploaded file
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        raw_image = Image.open(io.BytesIO(content))
        raw_image.verify()
        raw_image = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format or corrupted file: {exc}",
        )

    # 2. Preprocess user image (deskewing, polarity correction, stroke estimation)
    try:
        query_crop, deskewed_full, estimated_weight, detected_skew = preprocess_user_image(raw_image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error preprocessing image: {exc}",
        )

    # 3. Resolve text to compare (user override or OCR)
    target_text = ""
    if override_text and override_text.strip():
        target_text = override_text.strip()
    else:
        target_text = extract_text_from_image(deskewed_full)
        if not target_text:
            target_text = extract_text_from_image(query_crop)

    if not target_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No text could be detected in the image. Please enter the word in the 'Detected Text' field and try again.",
        )

    full_crop = query_crop

    model: CLIPModel = app.state.model
    processor: CLIPProcessor = app.state.processor
    device: torch.device = app.state.device

    # 4. STAGE 1: Coarse Typographic Classification
    valid_categories = {"sans-serif", "serif", "monospace", "handwriting", "display"}
    if category_override and category_override.lower() in valid_categories:
        detected_category = category_override.lower()
        category_confidence = 1.0
        category_probabilities = {detected_category: 1.0}
    else:
        detected_category, category_confidence, category_probabilities = classify_typography(
            full_crop, model, processor, device
        )

    logger.info(
        f"Stage 1 Typographic Classification: '{detected_category}' ({category_confidence:.2%} confidence) | "
        f"Estimated Weight: {estimated_weight} | Deskew: {detected_skew}°"
    )

    # 5. Select Candidate Families from Catalog
    candidate_families: list[str] = []

    # 5a. Always guarantee inclusion of universal workhorse core fonts
    candidate_families.extend(UNIVERSAL_CORE_FONTS)

    # 5b. Add primary category candidates
    primary_fonts = [f["family"] for f in catalog if f["category"] == detected_category]
    candidate_families.extend(primary_fonts[:30])

    # 5c. Multi-category soft pooling based on category probabilities
    sorted_probs = sorted(category_probabilities.items(), key=lambda x: x[1], reverse=True)
    for cat, prob in sorted_probs:
        if cat != detected_category and prob >= 0.10:
            secondary_fonts = [f["family"] for f in catalog if f["category"] == cat]
            candidate_families.extend(secondary_fonts[:10])

    # Deduplicate candidate family names preserving order
    candidate_families = list(dict.fromkeys(candidate_families))

    # 6. STAGE 2: Parallel In-Memory Font Subsetting
    fetched_candidates: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_fam = {
            executor.submit(fetch_family_subsets, fam, target_text, estimated_weight): fam
            for fam in candidate_families
        }
        for future in concurrent.futures.as_completed(future_to_fam):
            fam = future_to_fam[future]
            try:
                subsets = future.result()
                fetched_candidates.extend(subsets)
            except Exception as exc:
                logger.warning(f"Error fetching font subset for {fam}: {exc}")

    candidate_defs: list[dict[str, Any]] = []
    metadata_candidates: list[dict[str, Any]] = []
    source = "cloud_subset"

    if len(fetched_candidates) >= 5:
        for cand in fetched_candidates:
            cat = next((f["category"] for f in catalog if f["family"] == cand["family"]), detected_category)
            candidate_defs.append({
                "bytes": cand["bytes"],
                "path": None,
                "family": cand["family"],
                "variant": cand["variant"],
            })
            metadata_candidates.append({
                "family": cand["family"],
                "variant": cand["variant"],
                "category": cat,
                "google_fonts_url": f"https://fonts.google.com/specimen/{cand['family'].replace(' ', '+')}",
            })
    else:
        # Fallback to local ./fonts if cloud subsetting fails (offline resilient)
        source = "local_fallback"
        logger.warning(f"Cloud subsetting yielded only {len(fetched_candidates)} variants. Falling back to local font pool.")
        for item in font_library:
            candidate_defs.append({
                "bytes": None,
                "path": item["font_path"],
                "family": item["family"],
                "variant": item["variant"],
            })
            metadata_candidates.append({
                "family": item["family"],
                "variant": item["variant"],
                "category": item.get("category", detected_category),
                "google_fonts_url": item["google_fonts_url"],
            })

    if not candidate_defs:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate font candidate renders.",
        )

    dino_model: torch.nn.Module | None = getattr(app.state, "dino_model", None)
    dino_transform: Any | None = getattr(app.state, "dino_transform", None)

    def _render_def(defn: dict[str, Any], subtext: str) -> Image.Image:
        if defn["bytes"] is not None:
            return render_font_sample_from_bytes(defn["bytes"], subtext, font_size=72)
        return render_font_sample(defn["path"], subtext, font_size=72)

    # 7. Fine-Grained Matching: DINOv2 whole-word pre-rank, then per-glyph refinement.
    try:
        if dino_model is None or dino_transform is None:
            # Fallback: single-resolution CLIP cosine if DINOv2 is unavailable.
            q_inputs = processor(images=query_crop, return_tensors="pt")
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            with torch.no_grad():
                q_out = model.get_image_features(**q_inputs)
            q_feat = q_out.image_embeds if hasattr(q_out, "image_embeds") else q_out.pooler_output
            q_feat = q_feat / q_feat.norm(p=2, dim=-1, keepdim=True)

            cand_imgs = [_render_def(d, target_text) for d in candidate_defs]
            c_inputs = processor(images=cand_imgs, return_tensors="pt")
            c_inputs = {k: v.to(device) for k, v in c_inputs.items()}
            with torch.no_grad():
                c_out = model.get_image_features(**c_inputs)
            c_feats = c_out.image_embeds if hasattr(c_out, "image_embeds") else c_out.pooler_output
            c_feats = c_feats / c_feats.norm(p=2, dim=-1, keepdim=True)
            sims = (c_feats @ q_feat.T).squeeze(1).cpu().numpy()
        else:
            # Query: whole-word + per-glyph ink crops composed into one vector
            q_whole = embed_batch_dino(dino_model, dino_transform, [query_crop], device)[0]
            query_glyphs = extract_query_glyph_crops(deskewed_full)
            if query_glyphs:
                q_feat = compose_fine_feature(
                    embed_batch_dino(dino_model, dino_transform, [query_crop] + query_glyphs, device)
                )
            else:
                q_feat = q_whole

            # Whole-word pass across every candidate (coarse, fast)
            whole_feats = embed_batch_dino(
                dino_model, dino_transform,
                [_render_def(d, target_text) for d in candidate_defs],
                device,
            )
            whole_sims = whole_feats @ q_whole
            c_feats = whole_feats.copy()

            # Per-glyph refinement for the top-ranked candidates only
            refine_order = np.argsort(whole_sims)[::-1][: min(FINE_STAGE_LIMIT, len(candidate_defs))]
            refined_parts: dict[int, list[Image.Image]] = {}
            refined_images: list[Image.Image] = []
            for pos in refine_order:
                p = int(pos)
                parts = render_candidate_parts(lambda t, _p=p: _render_def(candidate_defs[_p], t), target_text)
                refined_parts[p] = parts
                refined_images.extend(parts)

            if refined_images:
                refined_feats = embed_batch_dino(dino_model, dino_transform, refined_images, device)
                offset = 0
                for pos in refine_order:
                    p = int(pos)
                    seg = refined_feats[offset : offset + len(refined_parts[p])]
                    offset += len(refined_parts[p])
                    c_feats[p] = compose_fine_feature(seg)

            sims = c_feats @ q_feat
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting visual embeddings: {exc}",
        )

    # 8. Deduplicate and keep best score per font family
    sorted_indices = np.argsort(sims)[::-1]
    family_best: dict[str, dict[str, Any]] = {}

    for rank_idx in sorted_indices:
        raw_score = round(float(sims[rank_idx]), 4)
        calibrated_score = calibrate_score(raw_score)
        meta = metadata_candidates[rank_idx]
        fam = meta["family"]

        if fam not in family_best or calibrated_score > family_best[fam]["similarity_score"]:
            family_best[fam] = {
                "family": fam,
                "variant": meta["variant"],
                "category": meta["category"],
                "similarity_score": calibrated_score,
                "raw_score": raw_score,
                "google_fonts_url": meta["google_fonts_url"],
            }

    top_families = sorted(
        family_best.values(),
        key=lambda x: x["similarity_score"],
        reverse=True,
    )[:top_k]

    print(f"\n--- Dynamic Match Results (Category: {detected_category}, Weight: {estimated_weight}, Skew: {detected_skew}°, Source: {source}) ---")
    matches: list[FontMatch] = []
    for candidate in top_families:
        pairings = get_font_pairings(candidate["family"], candidate["category"])
        print(f"Candidate: {candidate['family']} ({candidate['variant']}) [{candidate['category']}] | Raw: {candidate['raw_score']:.4f} | Calibrated: {candidate['similarity_score']:.4f} | Pairings: {', '.join(pairings)}")
        matches.append(
            FontMatch(
                family=candidate["family"],
                variant=candidate["variant"],
                category=candidate["category"],
                similarity_score=candidate["similarity_score"],
                raw_score=candidate["raw_score"],
                google_fonts_url=candidate["google_fonts_url"],
                recommended_pairings=pairings,
            )
        )

    return MatchResponse(
        detected_text=target_text,
        detected_category=detected_category,
        category_confidence=category_confidence,
        category_probabilities=category_probabilities,
        estimated_weight=estimated_weight,
        detected_skew=detected_skew,
        matches=matches,
        source=source,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
