import torch
import open_clip
import chromadb
from PIL import Image
from pathlib import Path
from moondream.moon import analyze_image, build_pdf
# from moondream.moon_ollama import analyze_image, build_pdf
import signal
import sys
import time
import datetime
import hashlib
import json
import numpy as np

# ── GPU MONITORING ────────────────────────────────────────────────────────────
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
    from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
    nvmlInit()
    _nvml_handle = nvmlDeviceGetHandleByIndex(0)
    HAS_NVML     = True
except Exception:
    _nvml_handle = None
    HAS_NVML     = False

def gpu_stats() -> str:
    if not HAS_NVML:
        return ""
    try:
        gpu  = nvmlDeviceGetUtilizationRates(_nvml_handle)
        mem  = nvmlDeviceGetMemoryInfo(_nvml_handle)
        return f"GPU:{gpu.gpu}% VRAM:{mem.used // 1024**2}MB"
    except Exception:
        return ""

# ── LOG SETUP ─────────────────────────────────────────────────────────────────
log_file = f"pipeline_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log(msg: str, also_print: bool = True):
    ts  = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def log_separator(char: str = "-", width: int = 60):
    log(char * width)

# ── HEADER ────────────────────────────────────────────────────────────────────
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"Pipeline started : {datetime.datetime.now()}\n")
    f.write("=" * 60 + "\n")

# ── SYSTEM INFO ───────────────────────────────────────────────────────────────
log(f"Torch version    : {torch.__version__}")
log(f"CUDA available   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    log(f"GPU              : {torch.cuda.get_device_name(0)}")
    log(f"CUDA version     : {torch.version.cuda}")
    log(f"VRAM             : {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB")
else:
    log("Running on CPU")

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device selected  : {device}")

# ── CONFIG ────────────────────────────────────────────────────────────────────
user_query  = "person with gun"
THRESHOLD   = 0.20
TOP_K       = 50
CACHE_FILE  = "./query_cache.json"

log_separator("=")
log(f"Query            : '{user_query}'")
log(f"Threshold        : {THRESHOLD}")
log(f"Top K            : {TOP_K}")
log_separator("=")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
log("Loading CLIP model...", also_print=True)
t0 = time.time()

tokenizer = open_clip.get_tokenizer("ViT-B-32")
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device)
model.eval()

model_load_time = time.time() - t0
log(f"Model loaded     : {model_load_time:.2f}s  |  device: {next(model.parameters()).device}")

# ── CONNECT DB ────────────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path="./vectordb")
try:
    collection = client.get_collection("images")
except Exception:
    collection = client.create_collection(
        "images", metadata={"hnsw:space": "cosine"}
    )
log(f"DB connected     : {collection.count()} images indexed")

# ── QUERY HELPERS ─────────────────────────────────────────────────────────────
def load_cache() -> dict:
    if Path(CACHE_FILE).exists():
        raw = json.loads(Path(CACHE_FILE).read_text())
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    return {}

def save_cache(cache: dict):
    Path(CACHE_FILE).write_text(
        json.dumps({k: v.tolist() for k, v in cache.items()})
    )

query_cache = load_cache()
log(f"Query cache      : {len(query_cache)} embeddings loaded")

def get_batch_embeddings(queries: list) -> np.ndarray:
    keys     = [hashlib.md5(q.encode()).hexdigest() for q in queries]
    uncached = [(i, q) for i, (q, k) in enumerate(zip(queries, keys)) if k not in query_cache]

    if uncached:
        indices, texts = zip(*uncached)
        tokens = tokenizer(list(texts)).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.cpu().float().numpy()
        for idx, emb in zip(indices, embs):
            query_cache[keys[idx]] = emb
        save_cache(query_cache)

    return np.array([query_cache[k] for k in keys], dtype=np.float32)

def get_expanded_embedding(query: str) -> np.ndarray:
    """Average 4 phrasing variants for better CLIP recall."""
    variants = [
        query,
        f"a photo of {query}",
        f"an image showing {query}",
        f"a picture of {query}",
    ]
    embs     = get_batch_embeddings(variants)
    mean_emb = embs.mean(axis=0)
    return (mean_emb / np.linalg.norm(mean_emb)).astype(np.float32)

def dynamic_threshold(distances: list, floor: float = 0.15) -> float:
    confidences = [1 - d for d in distances]
    return max(sum(confidences) / len(confidences), floor)

# ── CLIP QUERY ────────────────────────────────────────────────────────────────
log_separator()
log("CLIP QUERY")
log_separator()

t0  = time.time()
emb = get_expanded_embedding(user_query)
embed_time = time.time() - t0
log(f"Text embed time  : {embed_time*1000:.1f}ms  (4 variants expanded + averaged)")

t0 = time.time()
results = collection.query(
    query_embeddings=[emb.tolist()],
    n_results=TOP_K,
    include=["distances", "metadatas"]
)
db_time = time.time() - t0
log(f"DB query time    : {db_time*1000:.1f}ms")
log(f"Total query time : {(embed_time + db_time)*1000:.1f}ms  {gpu_stats()}")



# Dynamic threshold
dyn_threshold = dynamic_threshold(results["distances"][0])
effective_threshold = max(THRESHOLD, dyn_threshold)
log(f"Dynamic threshold: {dyn_threshold:.4f}  |  Effective: {effective_threshold:.4f}")

# ── RESULTS TABLE ─────────────────────────────────────────────────────────────
log_separator()

header = f"  {'FILENAME':<35} {'SCORE':>8}  {'STATUS'}"
log(header)
log(f"  {'-'*52}")

selected_images = []
for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
    similarity = round(1 - dist, 4)
    filename   = Path(meta["path"]).name
    passed     = similarity >= effective_threshold
    icon       = "✅" if passed else "❌"
    status     = "PASS" if passed else "SKIP"
    log(f"  {filename:<35} {similarity:>8.4f}  {icon} {status}")
    if passed:
        selected_images.append({"path": meta["path"], "similarity": similarity})

log_separator()
log(f"Selected {len(selected_images)} / {len(results['metadatas'][0])} images  (threshold: {effective_threshold:.4f})")
log_separator()

# ── GRACEFUL STOP ─────────────────────────────────────────────────────────────
analysis_results = []

def save_and_exit(sig=None, frame=None):
    log(f"\n⚠️  Interrupted! Saving {len(analysis_results)} analyzed images to PDF...")
    if analysis_results:
        output_pdf = "./output/moondream_report_partial.pdf"
        Path("./output").mkdir(exist_ok=True)
        build_pdf(analysis_results, output_pdf)
        log(f"✅ Partial PDF saved: {output_pdf}")
    else:
        log("❌ No images analyzed yet, nothing to save.")
    sys.exit(0)

signal.signal(signal.SIGINT,  save_and_exit)
signal.signal(signal.SIGTERM, save_and_exit)

# ── MOONDREAM ANALYSIS ────────────────────────────────────────────────────────
log("MOONDREAM ANALYSIS")
log_separator()

total_images     = len(selected_images)
moondream_start  = time.time()
times            = []

for i, item in enumerate(selected_images, 1):
    img_path   = item["path"]
    similarity = item["similarity"]

    t0       = time.time()
    analysis = analyze_image(img_path)
    elapsed  = time.time() - t0
    times.append(elapsed)

    # ETA
    avg_time     = sum(times) / len(times)
    remaining    = total_images - i
    eta_secs     = avg_time * remaining
    eta_str      = f"{int(eta_secs // 60)}m{int(eta_secs % 60)}s"

    log(
        f"  [{i:>3}/{total_images}] {Path(img_path).name:<35} "
        f"score:{similarity:.4f}  time:{elapsed:.2f}s  ETA:{eta_str}  {gpu_stats()}"
    )

    analysis_results.append({
        "filename":   Path(img_path).name,
        "path":       img_path,
        "similarity": similarity,
        "analysis":   analysis,
    })

# ── MOONDREAM SUMMARY ─────────────────────────────────────────────────────────
moondream_total = time.time() - moondream_start
avg             = moondream_total / total_images if total_images else 0

log_separator()
log(f"Moondream done   : {total_images} images  {gpu_stats()}")
log(f"Total time       : {moondream_total:.1f}s")
log(f"Avg per image    : {avg:.2f}s")
log(f"Throughput       : {total_images / moondream_total:.1f} img/s" if moondream_total > 0 else "Throughput: N/A")

# ── PDF ───────────────────────────────────────────────────────────────────────
log_separator()
log("Generating PDF...")
t0 = time.time()

Path("./output").mkdir(exist_ok=True)
output_pdf = "./output/moondream_report.pdf"
build_pdf(analysis_results, output_pdf)

pdf_time = time.time() - t0
log(f"PDF saved        : {output_pdf}  ({pdf_time:.2f}s)")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
log_separator("=")
log("PIPELINE COMPLETE")
log(f"  Query          : '{user_query}'")
log(f"  DB images      : {collection.count()}")
log(f"  CLIP results   : {len(results['metadatas'][0])}")
log(f"  Selected       : {len(selected_images)}")
log(f"  Analyzed       : {len(analysis_results)}")
log(f"  Model load     : {model_load_time:.2f}s")
log(f"  CLIP query     : {(embed_time + db_time)*1000:.1f}ms")
log(f"  Moondream      : {moondream_total:.1f}s  ({avg:.2f}s/img)")
log(f"  PDF generation : {pdf_time:.2f}s")
log(f"  Log saved      : {log_file}")
log_separator("=")