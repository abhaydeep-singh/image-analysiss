import torch
import open_clip
import chromadb
from PIL import Image
from pathlib import Path
from moondream.moon import analyze_image, build_pdf
import signal
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
user_query = "person with gun"
THRESHOLD = 0.20
tokenizer = open_clip.get_tokenizer("ViT-B-32")
client = chromadb.PersistentClient(path="./vectordb")

# load model
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(device)

try:
    collection = client.get_collection("images")
except:
    collection = client.create_collection(
        "images",
        metadata={"hnsw:space": "cosine"}
    )

text_tokens = tokenizer(user_query).to(device)

with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)

text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
query_embedding = text_embeddings.mean(dim=0, keepdim=True)

results = collection.query(
    query_embeddings=[query_embedding.cpu().numpy()[0]],
    n_results=50,
    include=["distances", "metadatas"]
)

# Review all results with confidence scores
print(f"\n{'='*55}")
print(f"  Query: '{user_query}'  |  Threshold: {THRESHOLD}")
print(f"{'='*55}")
print(f"  {'FILENAME':<35} {'SCORE':>8}  {'STATUS'}")
print(f"  {'-'*50}")

selected_images = []
for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
    similarity = round(1 - dist, 4)
    filename = Path(meta["path"]).name
    status = "✅ PASS" if similarity >= THRESHOLD else "❌ SKIP"
    print(f"  {filename:<35} {similarity:>8.4f}  {status}")
    if similarity >= THRESHOLD:
        selected_images.append({"path": meta["path"], "similarity": similarity})

print(f"{'='*55}")
print(f"  Selected {len(selected_images)} / {len(results['metadatas'][0])} images\n")


# Graceful stop if needed
analysis_results = []

def save_and_exit(sig=None, frame=None):
    print(f"\n\n⚠️  Interrupted! Saving {len(analysis_results)} analyzed images to PDF...")
    if analysis_results:
        output_pdf = "./output/moondream_report_partial.pdf"
        Path("./output").mkdir(exist_ok=True)
        build_pdf(analysis_results, output_pdf)
        print(f"✅ Partial PDF saved: {output_pdf}")
    else:
        print("❌ No images analyzed yet, nothing to save.")
    sys.exit(0)

signal.signal(signal.SIGINT, save_and_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, save_and_exit)  # kill command


# Moondream analysis on selected images only 
for item in selected_images:
    img_path = item["path"]
    similarity = item["similarity"]
    print(f"Analyzing [{similarity:.4f}]: {img_path}")
    analysis = analyze_image(img_path)
    analysis_results.append({
        "filename": Path(img_path).name,
        "path": img_path,
        "similarity": similarity,
        "analysis": analysis,
    })

# Generate PDF report 
Path("./output").mkdir(exist_ok=True)
output_pdf = "./output/moondream_report.pdf"
build_pdf(analysis_results, output_pdf)
print(f"\nPDF saved: {output_pdf}")