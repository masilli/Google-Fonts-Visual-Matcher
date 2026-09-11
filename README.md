# Google Fonts Visual Matcher 🔍🔤

An intelligent, locally hosted typographic visual identification engine that identifies fonts from images and screenshots using **OpenAI CLIP embeddings** and **dynamic on-the-fly candidate rendering**.

Unlike traditional font matchers that rely on pre-indexed static glyph strings (which introduce severe lexical and character-frequency bias), this engine extracts the exact text from the uploaded image via **Tesseract OCR**, dynamically streams candidate micro-subsets containing **only** those characters from Google Fonts, and performs pure shape-based typographic cosine similarity ranking.

---

## ✨ Key Features

- **🌐 1,581 Google Fonts Catalog & On-Demand Cloud Subsetting:**
  - Dynamic in-memory font streaming via Google Fonts CSS2 subsetting API (`io.BytesIO` in RAM).
  - Fetches tiny ~2–3 KB micro-fonts per candidate on the fly without needing gigabytes of TTF files on disk.
  - Multi-tiered caching: High-speed RAM LRU cache + persistent SQLite disk cache (`font_cache.db`).
  - Offline fallback: Includes 79 curated local workhorse fonts.

- **📐 Computer Vision & Preprocessing Pipeline:**
  - **Automatic Deskewing:** Pure NumPy horizontal projection profile variance corrects text tilt from -12° to +12° at 0.5° resolution in <25ms.
  - **Optical Stroke Weight Estimation:** Measures horizontal run-lengths across vertical stems to estimate CSS weights (Light 300 to ExtraBold 800) for exact weight parity matching.
  - **Contrast-Preserving Normalization:** Retains anti-aliased edge boundaries and terminal cuts without harsh binarization dilation.
  - **Low-Res Upscaling:** Intelligently upscales small crops (<150px) with subtle unsharp masking to recover crisp character terminals.

- **🧠 Multi-Stage Typographic Matching:**
  - **Stage 1 (Coarse Filtering):** Zero-shot CLIP classification filters into macro categories (`sans-serif`, `serif`, `display`, `handwriting`, `monospace`).
  - **Soft Multi-Category Pooling:** Retains proportional candidate representation for borderline typography.
  - **Guaranteed Universal Core:** Always includes the top ubiquitous fonts (Inter, Roboto, Open Sans, Montserrat, etc.) to guarantee popular workhorses are never gated out.
  - **Stage 2 (Fine Vector Ranking):** Batch embeds rendered candidate glyphs with CLIP ViT-B/32 on Apple Silicon (MPS) or CUDA/CPU.

- **🎨 Rich Designer UI & Comparison Tools:**
  - **Interactive ROI Cropper (Cropper.js):** Zoom, pan, tight bounding box guides, and 90° rotation to isolate text lines from complex websites and banners.
  - **Side-by-Side Comparison Strip:** Direct, always-visible comparison of the user's original image next to the matched Google Font.
  - **Stateful Glyph Diff Inspector:**
    - **Ghost Overlay Mode:** Layers the user's ink (Neon Cyan) over the candidate font (Coral) using screen compositing—perfect matches glow violet/white.
    - **Side-by-Side Mode:** Stacks the user's ink and the candidate font at matching optical heights.
    - **Interactive Alignment:** Click-and-drag directly on the canvas to slide letterforms, or use 2px nudge arrow buttons (←, →, ↑, ↓).
    - **Size & Blend Controls:** Scale the suggested font from 50% to 180% and adjust ink/font blend opacity in real time.
  - **Curated Typography Pairings:** 2–3 harmonious complementary font suggestions for every matched family.
  - **1-Click Code Export:** Copy `<link>`, `@import`, Tailwind CSS configuration, or standard CSS snippets instantly.

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Tesseract OCR**:
  - **macOS (Homebrew):** `brew install tesseract`
  - **Ubuntu / Debian:** `sudo apt-get install tesseract-ocr`
  - **Windows:** Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/masilli/Google-Fonts-Visual-Matcher.git
cd Google-Fonts-Visual-Matcher

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run the Application

Start the local FastAPI server:

```bash
python3 main.py
```

Then open your browser to **`http://localhost:8000`**.

---

## 🛠 Project Structure

```
├── main.py                     # FastAPI backend, vision pipeline, CLIP matcher, and cloud subsetter
├── index.html                  # Modern Tailwind CSS frontend, ROI cropper & glyph diff inspector
├── google_fonts_catalog.json   # Catalog metadata of 1,581 Google Fonts
├── requirements.txt            # Python dependencies
├── metadata.json               # Offline fallback font metadata
├── download_fonts.py           # Optional utility to download offline fallback fonts
├── fonts/                      # Curated offline fallback TTF fonts
├── test_improvements.py        # Automated test suite for vision & API pipelines
└── .gitignore                  # Git ignore rules
```

---

## 🧪 Testing

Run the automated verification suite to test deskewing, weight estimation, SQLite persistence, and API matching:

```bash
source .venv/bin/activate
python3 test_improvements.py
```

---

## 📄 License

This project is licensed under the MIT License. All Google Fonts loaded or included are governed by their respective open-source licenses (OFL, Apache 2.0, or Ubuntu Font License).
