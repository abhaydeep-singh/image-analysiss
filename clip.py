import datetime
import psutil
import time
import os
import torch
import open_clip
import chromadb
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE          = 384      # GTX 1660 handles larger batches well in FP32
FOLDER              = "./assets"
DB_PATH             = "./vectordb"
VALID_EXT           = (".png", ".jpg", ".jpeg")
USE_FP16            = False    # GTX 1650/1660 has no Tensor Cores — FP16 is slower
SKIP_EXISTING_CHECK = True    # True = skip DB lookup for already-indexed images (faster start for large sets)
# ─────────────────────────────────────────────────────────────────────────────

# ── GPU monitoring ────────────────────────────────────────────────────────────
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
    from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
    HAS_NVML    = True
except Exception:
    nvml_handle = None
    HAS_NVML    = False

# ── Device ────────────────────────────────────────────────────────────────────
print("Torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU           :", torch.cuda.get_device_name(0))
    print("VRAM          :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("Device        :", device)

# ── Log file ──────────────────────────────────────────────────────────────────
log_file = f"index_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
fp16_str = "FP16" if (USE_FP16 and device == "cuda") else "FP32"

with open(log_file, "w") as f:
    f.write(f"Indexing started : {datetime.datetime.now()}\n")
    f.write(f"Device           : {device} ({fp16_str})\n")
    f.write(f"Batch size       : {BATCH_SIZE}\n")
    f.write(f"Folder           : {FOLDER}\n")
    f.write("-" * 95 + "\n")
    f.write("progress | decode | embed | db | total | CPU% RAM% GPU% VRAM | ETA\n")
    f.write("-" * 95 + "\n")


def log_usage(progress, decode_t, embed_t, db_t, total_t, eta_str="--"):
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    if HAS_NVML:
        gpu_info = nvmlDeviceGetUtilizationRates(nvml_handle)
        mem_info = nvmlDeviceGetMemoryInfo(nvml_handle)
        hw_str   = f"CPU:{cpu:.0f}% RAM:{ram:.0f}% GPU:{gpu_info.gpu}% VRAM:{mem_info.used // 1024**2}MB"
    else:
        hw_str   = f"CPU:{cpu:.0f}% RAM:{ram:.0f}% GPU:N/A"

    line = (
        f"{progress} | "
        f"decode:{decode_t:.2f}s embed:{embed_t:.2f}s db:{db_t:.2f}s total:{total_t:.2f}s | "
        f"{hw_str} | ETA:{eta_str}\n"
    )
    print(line.strip())
    with open(log_file, "a") as lf:
        lf.write(line)


# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading CLIP model...", end=" ", flush=True)
t0 = time.time()

model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model     = model.to(device)
model.eval()
torch.backends.cudnn.benchmark = True  # auto-tunes fastest conv algo for your GPU

if USE_FP16 and device == "cuda":
    model = model.half()
    print(f"✅ FP16 ({time.time()-t0:.1f}s)")
else:
    print(f"✅ FP32 ({time.time()-t0:.1f}s)")

# Warmup with real batch size — forces cuDNN to benchmark NOW
# so batch 1 doesn't pay the 20s penalty
print("Warming up (cuDNN benchmarking)...", end=" ", flush=True)
with torch.no_grad():
    dummy = torch.zeros(BATCH_SIZE, 3, 224, 224).to(device)
    if USE_FP16 and device == "cuda":
        dummy = dummy.half()
    model.encode_image(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
print("✅ done")


# ── Chroma DB ─────────────────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    "images",
    metadata={"hnsw:space": "cosine"}
)
existing_ids = set(collection.get(include=[])["ids"]) if not SKIP_EXISTING_CHECK else set()
print(f"Already indexed : {len(existing_ids)} images {'(check skipped)' if SKIP_EXISTING_CHECK else ''}")


# ── Validate images ───────────────────────────────────────────────────────────
print("Scanning folder...", end=" ", flush=True)
valid_files = []

for file in os.listdir(FOLDER):
    if not file.lower().endswith(VALID_EXT):
        continue
    if file in existing_ids:
        continue
    path = os.path.join(FOLDER, file)
    try:
        if os.path.getsize(path) == 0:
            continue
        with Image.open(path) as img:
            img.verify()
        valid_files.append((file, path))
    except (UnidentifiedImageError, OSError, ValueError):
        continue

print(f"✅ {len(valid_files)} new  |  {len(existing_ids)} already indexed")

if not valid_files:
    print("Nothing to do.")
    exit()


# ── Image opener ─────────────────────────────────────────────────────────────
def open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

with open(log_file, "a") as f:
    f.write(f"TurboJPEG        : not used (PIL sufficient at this batch size)\n")


# ── Prefetch pipeline ─────────────────────────────────────────────────────────
# CPU decodes next batch while GPU embeds current batch
def preprocess_batch(batch: list) -> tuple:
    """Runs on CPU thread. Returns pinned tensor for fast GPU transfer."""
    images, ids, metadatas = [], [], []
    t0 = time.time()
    for file, path in batch:
        try:
            img = preprocess(open_image(path)).unsqueeze(0)
            images.append(img)
            ids.append(file)
            metadatas.append({"path": path})
        except Exception:
            continue
    decode_time = time.time() - t0

    if not images:
        return None, [], [], 0.0

    tensor = torch.cat(images)
    if device == "cuda":
        tensor = tensor.pin_memory()  # faster CPU → GPU DMA, CUDA only
    return tensor, ids, metadatas, decode_time


batches    = [valid_files[i:i + BATCH_SIZE] for i in range(0, len(valid_files), BATCH_SIZE)]
prefetch_q = Queue(maxsize=3)  # buffer up to 3 batches ahead


def prefetch_worker():
    for b in batches:
        prefetch_q.put(preprocess_batch(b))
    prefetch_q.put(None)  # sentinel — signals end


# ── Main indexing loop ────────────────────────────────────────────────────────
total_batches  = len(batches)
time_per_batch = None
run_start      = time.time()

with ThreadPoolExecutor(max_workers=1) as prefetcher:
    prefetcher.submit(prefetch_worker)

    batch_num = 0
    while True:
        item = prefetch_q.get()
        if item is None:
            break

        batch_num += 1
        tensor, ids, metadatas, decode_time = item

        if tensor is None:
            continue

        batch_total_start = time.time()

        # non_blocking=True — GPU DMAs data without blocking CPU
        if USE_FP16 and device == "cuda":
            batch_tensor = tensor.to(device, non_blocking=True).half()
        else:
            batch_tensor = tensor.to(device, non_blocking=True)

        # Embed
        t1 = time.time()
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        embed_time = time.time() - t1

        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

        # DB insert
        t2 = time.time()
        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings.cpu().float().numpy().tolist(),
            metadatas=metadatas
        )
        db_time = time.time() - t2

        total_time     = time.time() - batch_total_start
        time_per_batch = total_time if time_per_batch is None else (time_per_batch * 0.8 + total_time * 0.2)

        batches_left = total_batches - batch_num
        eta_secs     = time_per_batch * batches_left
        eta_str      = f"{int(eta_secs // 60)}m{int(eta_secs % 60)}s"

        done     = min(batch_num * BATCH_SIZE, len(valid_files))
        progress = f"batch {batch_num}/{total_batches} ({done}/{len(valid_files)})"
        log_usage(progress, decode_time, embed_time, db_time, total_time, eta_str)


# ── Done ──────────────────────────────────────────────────────────────────────
total_run = time.time() - run_start
summary   = (
    f"\nIndexing complete | "
    f"total:{total_run:.1f}s | "
    f"throughput:{len(valid_files)/total_run:.1f} img/s | "
    f"DB total:{collection.count()}\n"
)
print(summary)
with open(log_file, "a") as f:
    f.write("-" * 95 + "\n")
    f.write(summary)