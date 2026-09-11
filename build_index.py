"""build_index.py

Offline indexing pipeline for Google Fonts Visual Matcher.
Renders text samples for TrueType fonts, extracts OpenAI CLIP visual feature
embeddings (512-d), L2-normalizes them, and builds a FAISS IndexFlatIP index along
with associated font metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any

# Prevent duplicate libomp crash on macOS when torch and faiss are loaded together
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_TEXT = "RaegyQG&"
CANVAS_SIZE = (224, 224)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512  # CLIP ViT-B/32 projection dimension


def get_device(preferred_device: str | None = None) -> torch.device:
    """Select the best available compute device (CUDA, Apple MPS, or CPU)."""
    if preferred_device:
        return torch.device(preferred_device)
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


def normalize_text_crop(
    img: Image.Image,
    target_size: tuple[int, int] = CANVAS_SIZE,
    padding_ratio: float = 0.1,
) -> Image.Image:
    """Normalize text crop to preserve natural font aspect ratio and ink scale.
    
    1. Converts image to grayscale and locates tight bounding box around text ink
       using PIL's getbbox() (inverts if necessary so ink pixels are positive).
    2. Crops strictly to this bounding box.
    3. Creates a square white canvas sized max(width, height) + (2 * padding).
    4. Centers the cropped text on this square canvas.
    5. Resizes final padded square canvas to target_size using LANCZOS resampling.
    """
    # 1. Convert to grayscale
    gray = img.convert("L")
    arr = np.array(gray)

    # Invert if necessary so ink pixels are positive (dark text on light background)
    if np.mean(arr) > 128.0:
        ink_mask = ImageOps.invert(gray)
        crop_source = gray.convert("RGB")
    else:
        ink_mask = gray
        crop_source = ImageOps.invert(gray).convert("RGB")

    # Threshold ink_mask slightly to ignore subtle background compression noise
    thresh_mask = ink_mask.point(lambda p: 255 if p > 25 else 0)
    bbox = thresh_mask.getbbox() or ink_mask.getbbox() or (0, 0, gray.width, gray.height)

    # 2. Crop strictly to bounding box
    cropped = crop_source.crop(bbox)
    cw, ch = cropped.size

    # 3. Create square white canvas: max(width, height) + (2 * padding)
    max_side = max(cw, ch)
    padding = max(int(max_side * padding_ratio), 1)
    canvas_dim = max_side + (2 * padding)

    canvas = Image.new("RGB", (canvas_dim, canvas_dim), color=(255, 255, 255))

    # 4. Center cropped text on square canvas
    offset_x = (canvas_dim - cw) // 2
    offset_y = (canvas_dim - ch) // 2
    canvas.paste(cropped, (offset_x, offset_y))

    # 5. Resize final padded square canvas to target_size using LANCZOS
    return canvas.resize(target_size, Image.Resampling.LANCZOS)


def render_font_sample(
    font_path: Path | str,
    text: str = DEFAULT_SAMPLE_TEXT,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
    padding: int = 16,
) -> Image.Image:
    """Render a text string onto a white canvas with black text,
    then normalize the text crop to preserve the natural font aspect ratio.
    """
    canvas_w, canvas_h = canvas_size
    avail_w = max(canvas_w - 2 * padding, 10)
    avail_h = max(canvas_h - 2 * padding, 10)
    font_file = str(font_path)

    # Initial test render to gauge aspect ratio and font metric scaling
    initial_font_size = 48
    font = ImageFont.truetype(font_file, size=initial_font_size)
    dummy_img = Image.new("RGB", (1, 1), (255, 255, 255))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Calculate scale factor to comfortably fill canvas while respecting margins
    scale = min(avail_w / max(text_w, 1), avail_h / max(text_h, 1))
    target_font_size = max(int(initial_font_size * scale), 8)

    # Re-instantiate font at target size
    final_font = ImageFont.truetype(font_file, size=target_font_size)
    canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    final_bbox = draw.textbbox((0, 0), text, font=final_font)
    w = final_bbox[2] - final_bbox[0]
    h = final_bbox[3] - final_bbox[1]

    # Center horizontally and vertically considering glyph ascender/descender offset
    pos_x = (canvas_w - w) // 2 - final_bbox[0]
    pos_y = (canvas_h - h) // 2 - final_bbox[1]

    draw.text((pos_x, pos_y), text, font=final_font, fill=(0, 0, 0))

    # Standardized text crop normalization: tight crop, square pad, resize LANCZOS
    return normalize_text_crop(canvas, target_size=canvas_size, padding_ratio=0.1)


def extract_font_metadata(font_path: Path) -> dict[str, str]:
    """Extract font family, variant, and constructed Google Fonts URL.
    
    Attempts to read TrueType metadata via Pillow FreeType font name tables,
    falling back to filename stem parsing when needed.
    """
    family: str = ""
    variant: str = "Regular"

    try:
        f = ImageFont.truetype(str(font_path), size=24)
        if hasattr(f, "getname"):
            name_info = f.getname()
            if name_info and len(name_info) >= 2:
                fam = name_info[0].strip() if name_info[0] else ""
                var = name_info[1].strip() if name_info[1] else ""
                if fam:
                    family = fam
                if var:
                    variant = var
    except Exception as e:
        logger.debug(f"Unable to read font name table from '{font_path.name}': {e}")

    # Fallback to filename parsing
    if not family:
        stem = font_path.stem
        if "-" in stem:
            parts = stem.split("-", 1)
            family = parts[0].strip()
            variant = parts[1].strip() or "Regular"
        elif "_" in stem:
            parts = stem.split("_", 1)
            family = parts[0].strip()
            variant = parts[1].strip() or "Regular"
        else:
            family = stem
            variant = "Regular"

    clean_family = family.replace("_", " ").strip()
    encoded_family = urllib.parse.quote_plus(clean_family)
    google_fonts_url = f"https://fonts.google.com/specimen/{encoded_family}"

    return {
        "family": clean_family,
        "font_family": clean_family,
        "variant": variant,
        "file_name": font_path.name,
        "google_fonts_url": google_fonts_url,
    }


def extract_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    device: torch.device,
) -> np.ndarray:
    """Pass image through CLIP vision encoder and return L2-normalized 512-d vector."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_image_features(**inputs)

    # Extract embedding tensor from CLIP output
    if isinstance(out, torch.Tensor):
        feat = out
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output
    elif hasattr(out, "image_embeds") and out.image_embeds is not None:
        feat = out.image_embeds
    else:
        feat = out[0]

    # L2-normalize feature tensor
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    embedding = np.ascontiguousarray(feat.cpu().numpy(), dtype=np.float32)  # shape: (1, 512)
    faiss.normalize_L2(embedding)
    return embedding


def build_index(
    fonts_dir: str | Path = "./fonts",
    index_path: str | Path = "fonts.index",
    metadata_path: str | Path = "metadata.json",
    sample_text: str = DEFAULT_SAMPLE_TEXT,
    preferred_device: str | None = None,
    model_name: str = CLIP_MODEL_NAME,
) -> tuple[int, int]:
    """Scans fonts directory, extracts CLIP embeddings, and creates FAISS index & metadata file."""
    fonts_directory = Path(fonts_dir)
    if not fonts_directory.exists():
        raise FileNotFoundError(f"Fonts directory '{fonts_directory.resolve()}' does not exist.")

    # Find all TrueType / OpenType font files
    font_files = sorted([
        p for p in fonts_directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".ttf", ".otf"}
    ])

    if not font_files:
        logger.warning(
            f"No font files (.ttf, .otf) found in '{fonts_directory.resolve()}'. "
            "Please add font files to the directory before building the index."
        )
        return 0, 0

    logger.info(f"Discovered {len(font_files)} font files in '{fonts_directory}'.")

    device = get_device(preferred_device)
    logger.info(f"Using compute device: {device}")
    model, processor = load_clip_model(model_name=model_name, device=device)

    embeddings_list: list[np.ndarray] = []
    metadata_list: list[dict[str, Any]] = []
    failed_count = 0

    for idx, font_path in enumerate(font_files, start=1):
        try:
            # 1. Render sample text (passes through normalize_text_crop)
            img = render_font_sample(font_path, text=sample_text, canvas_size=CANVAS_SIZE)

            # 2. Extract L2-normalized CLIP embedding (512-d)
            emb = extract_embedding(model, processor, img, device)

            # 3. Extract metadata
            meta = extract_font_metadata(font_path)
            meta["id"] = len(metadata_list)

            embeddings_list.append(emb)
            metadata_list.append(meta)

            if idx % 10 == 0 or idx == len(font_files):
                logger.info(f"Processed {idx}/{len(font_files)} fonts...")

        except Exception as err:
            failed_count += 1
            logger.error(
                f"Failed to process font '{font_path.name}' ({font_path}): {err}",
                exc_info=False,
            )

    if not embeddings_list:
        logger.error("No fonts were successfully processed. Index creation aborted.")
        return 0, failed_count

    # Stack into a continuous (N, 512) float32 matrix
    embeddings_matrix = np.ascontiguousarray(np.vstack(embeddings_list), dtype=np.float32)

    # Build FAISS IndexFlatIP with 512 dimensions (Inner Product with L2-normalized vectors = Cosine Similarity)
    logger.info(f"Building FAISS IndexFlatIP with dimension {EMBEDDING_DIM}...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_matrix)

    # Save index
    out_index_path = Path(index_path)
    logger.info(f"Saving FAISS index to '{out_index_path.resolve()}'...")
    faiss.write_index(index, str(out_index_path))

    # Save metadata JSON
    out_meta_path = Path(metadata_path)
    logger.info(f"Saving metadata to '{out_meta_path.resolve()}'...")
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Indexing complete! Successfully indexed {len(metadata_list)} fonts "
        f"({failed_count} failed)."
    )
    return len(metadata_list), failed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS visual search index for Google Fonts using OpenAI CLIP."
    )
    parser.add_argument(
        "--fonts-dir",
        type=str,
        default="./fonts",
        help="Directory containing .ttf/.otf files (default: ./fonts)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="fonts.index",
        help="Output FAISS index filepath (default: fonts.index)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="metadata.json",
        help="Output metadata JSON filepath (default: metadata.json)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_SAMPLE_TEXT,
        help=f"Sample text to render for font embeddings (default: '{DEFAULT_SAMPLE_TEXT}')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch compute device ('cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=CLIP_MODEL_NAME,
        help=f"Hugging Face CLIP model ID (default: '{CLIP_MODEL_NAME}')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        indexed, failed = build_index(
            fonts_dir=args.fonts_dir,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            sample_text=args.text,
            preferred_device=args.device,
            model_name=args.model_name,
        )
    except Exception as exc:
        logger.error(f"Fatal error during indexing: {exc}")
        sys.exit(1)
