import io
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fastapi.testclient import TestClient

import main
from main import (
    app,
    deskew_text_image,
    estimate_stroke_weight,
    db_get_font,
    db_put_font,
    cache_get_font,
    cache_put_font,
    get_font_pairings,
    init_cache_db,
    render_candidate_parts,
    render_font_sample_from_bytes,
    compose_fine_feature,
    CANVAS_SIZE,
)

def test_deskewing():
    print("Testing deskewing algorithm...")
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("fonts/Inter-Regular.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 25), "Typography Test", fill=(0, 0, 0), font=font)
    
    # Skew by 5 degrees clockwise
    tilted = img.rotate(5.0, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    corrected, angle = deskew_text_image(tilted)
    print(f"Detected tilt: {angle}° (expected ~ -5.0°)")
    assert abs(abs(angle) - 5.0) <= 1.0, f"Deskew angle {angle} was not within tolerance"
    print("✓ Deskewing verified.")

def test_stroke_weight_estimation():
    print("Testing stroke weight estimation...")
    # Regular
    img_reg = Image.new("RGB", (300, 80), color=(255, 255, 255))
    draw_reg = ImageDraw.Draw(img_reg)
    font_reg = ImageFont.truetype("fonts/Inter-Regular.ttf", 44)
    draw_reg.text((20, 15), "Identify", fill=(0, 0, 0), font=font_reg)
    weight_reg, ratio_reg = estimate_stroke_weight(img_reg)
    print(f"Regular weight: {weight_reg} (ratio: {ratio_reg})")

    # Bold
    img_bold = Image.new("RGB", (300, 80), color=(255, 255, 255))
    draw_bold = ImageDraw.Draw(img_bold)
    font_bold = ImageFont.truetype("fonts/Inter-Bold.ttf", 44)
    try:
        font_bold.set_variation_by_name("Bold")
    except Exception:
        pass
    draw_bold.text((20, 15), "Identify", fill=(0, 0, 0), font=font_bold)
    weight_bold, ratio_bold = estimate_stroke_weight(img_bold)
    print(f"Bold weight: {weight_bold} (ratio: {ratio_bold})")

    assert weight_reg in (300, 400), f"Expected 300 or 400 for Regular, got {weight_reg}"
    assert weight_bold in (600, 700, 800), f"Expected 600-800 for Bold, got {weight_bold}"
    assert ratio_bold > ratio_reg, f"Bold ratio ({ratio_bold}) should be > Regular ({ratio_reg})"
    print("✓ Stroke weight estimation verified.")

def test_sqlite_persistence():
    print("Testing SQLite persistence...")
    init_cache_db()
    test_family = "TestFont"
    test_weight = "700"
    test_text = "Hello"
    test_bytes = b"TrueTypeFontDummyBytes12345"

    cache_put_font(test_family, test_weight, test_text, test_bytes)
    # Clear RAM cache entry to force reading from disk
    key = (test_family.lower(), test_weight, test_text)
    if key in main._FONT_CACHE:
        del main._FONT_CACHE[key]

    recovered = cache_get_font(test_family, test_weight, test_text)
    assert recovered == test_bytes, "Recovered bytes from SQLite do not match original"
    print("✓ SQLite micro-font persistence verified.")

def test_font_pairings():
    print("Testing curated pairings...")
    inter_pairings = get_font_pairings("Inter", "sans-serif")
    assert len(inter_pairings) >= 2, "Inter pairings should have at least 2 fonts"
    print(f"Inter pairings: {inter_pairings}")

    serif_pairings = get_font_pairings("Playfair Display", "serif")
    assert len(serif_pairings) >= 2
    print(f"Playfair pairings: {serif_pairings}")
    print("✓ Font pairings verified.")

def test_compose_fine_feature():
    print("Testing fine feature composition...")
    rng = np.random.default_rng(7)
    parts = rng.normal(size=(5, 384)).astype(np.float32)
    parts /= np.linalg.norm(parts, axis=1, keepdims=True)
    combined = compose_fine_feature(parts)
    assert abs(float(np.linalg.norm(combined)) - 1.0) < 1e-5, "Output must be L2-normalized"
    single = compose_fine_feature(parts[:1])
    assert np.allclose(single, parts[0]), "Single-part input should pass through unchanged"
    print("✓ Fine feature composition verified.")

def test_render_candidate_parts():
    print("Testing per-glyph candidate rendering...")
    with open("fonts/Inter-Regular.ttf", "rb") as fh:
        data = fh.read()
    parts = render_candidate_parts(
        lambda t: render_font_sample_from_bytes(data, t, font_size=72),
        "Identify",
    )
    assert len(parts) >= 2, "Per-glyph rendering should yield whole-word + individual glyphs"
    for p in parts:
        assert p.size == CANVAS_SIZE, "All candidate parts must be canvas-resized"
    print(f"✓ Candidate per-glyph rendering verified ({len(parts)} parts).")

def test_api_e2e():
    print("Testing FastAPI /api/match end-to-end...")
    # Render test image
    img = Image.new("RGB", (320, 90), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("fonts/Inter-Bold.ttf", 44)
    draw.text((25, 20), "Identify", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with TestClient(app) as client:
        # Check health
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        print("Health status:", h_resp.json())

        # Test match
        response = client.post(
            "/api/match",
            files={"file": ("test.png", buf.getvalue(), "image/png")},
            data={"top_k": 3},
        )
        assert response.status_code == 200, f"Match request failed: {response.text}"
        data = response.json()
        print("Match response:")
        print(f"  Detected Text: {data['detected_text']}")
        print(f"  Detected Category: {data['detected_category']} (Conf: {data['category_confidence']})")
        print(f"  Estimated Weight: {data['estimated_weight']}")
        print(f"  Detected Skew: {data['detected_skew']}°")
        print(f"  Source: {data['source']}")
        print(f"  Matches: {len(data['matches'])}")
        for m in data['matches']:
            print(f"    - {m['family']} ({m['variant']}): Score {m['similarity_score']} | Pairings: {m['recommended_pairings']}")

        assert data["detected_text"].lower() == "identify"
        assert len(data["matches"]) > 0
        assert "estimated_weight" in data
        assert "detected_skew" in data
        assert all(len(m["recommended_pairings"]) > 0 for m in data["matches"])
        print("✓ API End-to-End verified successfully.")

if __name__ == "__main__":
    test_deskewing()
    test_stroke_weight_estimation()
    test_sqlite_persistence()
    test_font_pairings()
    test_compose_fine_feature()
    test_render_candidate_parts()
    test_api_e2e()
    print("\nALL VERIFICATION TESTS PASSED!")
