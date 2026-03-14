import moondream as md

# ── Load model ONCE globally ──────────────────────────────────────────────────
print("🔄 Loading Moondream model...")
_model = md.vl(model="moondream-2b-int8.mnt")
print("✅ Model ready\n")


# ── Image analysis ────────────────────────────────────────────────────────────
def analyze_image(image_path: str) -> str:
    try:
        image = PILImage.open(image_path)
        encoded = _model.encode_image(image)
        result = _model.query(encoded, PROMPT)["answer"]
        return result.strip()
    except Exception as e:
        return f"[Analysis failed: {str(e)}]"