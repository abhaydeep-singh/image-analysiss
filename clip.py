import torch
import open_clip
from PIL import Image, UnidentifiedImageError
import os
import chromadb

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    device = "cuda"
else:
    device = "cpu"

print("Device used:", device)

model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
print("Model device:", next(model.parameters()).device)

# Vector DB using :contentReference[oaicite:1]{index=1}
client = chromadb.PersistentClient(path="./vectordb")

collection = client.get_or_create_collection(
    "images",
    metadata={"hnsw:space": "cosine"}
)

folder = "./assets"

valid_ext = (".png", ".jpg", ".jpeg")



for file in os.listdir(folder):

    # 1️⃣ extension check
    if not file.lower().endswith(valid_ext):
        # print(f"Skipping non-image file: {file}")
        continue

    path = os.path.join(folder, file)

    try:
        # 2️⃣ check file size
        if os.path.getsize(path) == 0:
            # print(f"Skipping empty file: {file}")
            continue

        # 3️⃣ verify image integrity
        with Image.open(path) as img:
            img.verify()

        # reopen image after verify
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    except (UnidentifiedImageError, OSError, ValueError) as e:
        # print(f"Skipping corrupted image: {file} | Error: {e}")
        continue

    # Encode image using :contentReference[oaicite:2]{index=2}
    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    collection.add(
        ids=[file],
        embeddings=[embedding.cpu().numpy()[0]],
        metadatas=[{"path": path}]
    )

    # print(f"Indexed: {file}")

print("Images indexed successfully")