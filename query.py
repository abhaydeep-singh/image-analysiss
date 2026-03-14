import torch
import open_clip
import chromadb
import hashlib
import numpy as np
import json
import time
import threading
from pathlib import Path

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("Device:", device)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CACHE_FILE   = "./query_cache.json"
DB_PATH      = "./vectordb"
TOP_K        = 5
USE_FP16     = device == "cuda"   # FP16 only on GPU
USE_COMPILE  = device == "cuda" and hasattr(torch, "compile")
BATCH_SIZE   = 64 if device == "cuda" else 8
# ─────────────────────────────────────────────────────────────────────────────


# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading CLIP model...", end=" ", flush=True)
t0 = time.time()

model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model     = model.to(device)
model.eval()

# FP16 on GPU — 2-4x faster, half the VRAM
if USE_FP16:
    model = model.half()
    print(f"✅ loaded in FP16 ({time.time()-t0:.1f}s)")
else:
    print(f"✅ loaded ({time.time()-t0:.1f}s)")

# torch.compile — GPU only, big speedup after first call
if USE_COMPILE:
    print("Compiling model (one-time, ~30s)...", end=" ", flush=True)
    t0    = time.time()
    model = torch.compile(model, mode="reduce-overhead")
    print(f"✅ ({time.time()-t0:.1f}s)")

# Warmup
print("Warming up...", end=" ", flush=True)
t0 = time.time()
with torch.no_grad():
    dummy = tokenizer(["warmup"]).to(device)
    if USE_FP16:
        dummy = dummy
    model.encode_text(dummy)
print(f"✅ ({time.time()-t0:.2f}s)")

if device == "cuda":
    torch.cuda.synchronize()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e6:.0f} MB")


# ── Connect to DB ─────────────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("images")
print(f"\n✅ DB connected — {collection.count()} images indexed")


# ── Persistent disk cache ─────────────────────────────────────────────────────
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
print(f"💾 Cache: {len(query_cache)} embeddings loaded\n")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EMBEDDING — send many queries to GPU at once
# ─────────────────────────────────────────────────────────────────────────────
def get_batch_embeddings(queries: list) -> np.ndarray:
    keys      = [hashlib.md5(q.encode()).hexdigest() for q in queries]
    uncached  = [(i, q) for i, (q, k) in enumerate(zip(queries, keys)) if k not in query_cache]

    if uncached:
        indices, texts = zip(*uncached)

        # Process in chunks of BATCH_SIZE (GPU loves large batches)
        all_embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk  = list(texts[i:i + BATCH_SIZE])
            tokens = tokenizer(chunk).to(device)

            with torch.no_grad():
                embs = model.encode_text(tokens)
                embs = embs / embs.norm(dim=-1, keepdim=True)

            # Move to CPU only once per batch
            all_embs.append(embs.cpu().float().numpy())

        batch_result = np.concatenate(all_embs, axis=0)

        for idx, emb in zip(indices, batch_result):
            query_cache[keys[idx]] = emb
        save_cache(query_cache)

    return np.array([query_cache[k] for k in keys], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# QUERY EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
def get_expanded_embedding(query: str) -> np.ndarray:
    variants = [
        query,
        f"a photo of {query}",
        f"an image showing {query}",
        f"a picture of {query}",
    ]
    # All 4 variants encoded in ONE GPU batch call
    embs     = get_batch_embeddings(variants)
    mean_emb = embs.mean(axis=0)
    return (mean_emb / np.linalg.norm(mean_emb)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC DB LOOKUP — query Chroma while GPU works on next embedding
# ─────────────────────────────────────────────────────────────────────────────
class AsyncDBQuery:
    def __init__(self):
        self.result    = None
        self.thread    = None

    def start(self, embedding: np.ndarray, top_k: int = TOP_K):
        self.result = None
        self.thread = threading.Thread(
            target=self._run, args=(embedding, top_k)
        )
        self.thread.start()

    def _run(self, embedding, top_k):
        self.result = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            include=["distances", "metadatas"]
        )

    def wait(self):
        if self.thread:
            self.thread.join()
        return self.result


async_db = AsyncDBQuery()


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
def dynamic_threshold(distances: list, floor: float = 0.15) -> float:
    confidences = [1 - d for d in distances]
    return max(sum(confidences) / len(confidences), floor)


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = TOP_K) -> list:
    t0  = time.time()
    emb = get_expanded_embedding(query)

    results   = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=top_k,
        include=["distances", "metadatas"]
    )
    threshold = dynamic_threshold(results["distances"][0])
    elapsed   = (time.time() - t0) * 1000

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        confidence = 1 - dist
        output.append({
            "path":       meta["path"],
            "confidence": confidence,
            "above":      confidence >= threshold,
            "ms":         elapsed,
        })
    return output


def search_multi(queries: list, top_k: int = TOP_K) -> list:
    t0 = time.time()

    # Expand all queries first — one big GPU batch for everything
    all_variants = []
    for q in queries:
        all_variants += [q, f"a photo of {q}", f"an image showing {q}", f"a picture of {q}"]

    all_embs = get_batch_embeddings(all_variants)  # single GPU call

    # Reconstruct per-query averaged embeddings
    query_embs = []
    for i in range(len(queries)):
        chunk    = all_embs[i*4:(i+1)*4]
        mean_emb = chunk.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        query_embs.append(mean_emb)

    # Query DB for each
    all_scores = {}
    for emb in query_embs:
        results = collection.query(
            query_embeddings=[emb.tolist()],
            n_results=top_k * 2,
            include=["distances", "metadatas"]
        )
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            path       = meta["path"]
            confidence = 1 - dist
            if path not in all_scores or confidence > all_scores[path]["confidence"]:
                all_scores[path] = {"confidence": confidence, "path": path}

    elapsed = (time.time() - t0) * 1000
    ranked  = sorted(all_scores.values(), key=lambda x: x["confidence"], reverse=True)
    for r in ranked:
        r["ms"] = elapsed
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def print_results(results: list, label: str = ""):
    if label:
        print(f"\n{'=' * 65}")
        print(f"  {label}")
        print(f"{'=' * 65}")
    if results:
        print(f"  ⏱  {results[0].get('ms', 0):.1f}ms")
    for i, r in enumerate(results, 1):
        confidence = r["confidence"]
        path       = r.get("path", "")
        above      = r.get("above", True)
        status     = "✅" if above else "❌"
        bar        = "█" * int(confidence * 30)
        print(f"  {i}. {status} {confidence:.4f}  {bar}  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    results = search("person with gun")
    print_results(results, "Single: 'person with gun'")

    results = search_multi([
        "person with gun",
        "man holding weapon",
        "firearm pistol rifle",
        "armed person",
    ])
    print_results(results, "Multi-query fusion: weapons")

    print(f"\n{'=' * 65}")
    print("  INTERACTIVE  |  separate terms with | for multi-query")
    print(f"{'=' * 65}\n")

    while True:
        raw = input("🔍 Query: ").strip()
        if not raw or raw.lower() in ("quit", "q", "exit"):
            break

        terms = [t.strip() for t in raw.split("|") if t.strip()]

        if len(terms) > 1:
            results = search_multi(terms)
            print_results(results, f"Multi: {terms}")
        else:
            results = search(terms[0])
            print_results(results, f"'{terms[0]}'")