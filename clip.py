import torch
import open_clip
from PIL import Image, UnidentifiedImageError
import os
import chromadb

# Parameters
BATCH_SIZE = 10 #works on both CPU and GPU 
folder = "./assets"
valid_ext = (".png", ".jpg", ".jpeg")


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
model.eval()  #inference / testing mode for model

client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_or_create_collection(
    "images",
    metadata={"hnsw:space": "cosine"}
)
# Skipping already indexed ids
existing_ids = set(collection.get()["ids"])
print(f"Already indexed: {len(existing_ids)} images")

valid_files = []
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
        valid_files.append((file, path))
        # reopen image after verify

    except (UnidentifiedImageError, OSError, ValueError) as e:
        # print(f"Skipping corrupted image: {file} | Error: {e}")
        continue

print(f"Images Ready for Index: {len(valid_files)}")


# Batch Processing
for batch_start in range(0, len(valid_files), BATCH_SIZE):
    batch = valid_files[batch_start:batch_start + BATCH_SIZE]

    ids, embeddings, metadatas, images = [], [], [], []

    for file, path in batch:
        try:
            img = preprocess(Image.open(path)).unsqueeze(0)
            images.append(img)
            ids.append(file)
            metadatas.append({"path": path})
        except Exception:
            continue

    if not images:
        continue

    # stack into single tensor and encode all at once
    batch_tensor = torch.cat(images).to(device) # .cat joins (concatenates) multiple tensors together

    with torch.no_grad():
        batch_embeddings = model.encode_image(batch_tensor)

    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

    collection.add(
        ids=ids,
        embeddings=batch_embeddings.cpu().numpy().tolist(),
        metadatas=metadatas
    )

    print(f"  Indexed {batch_start + len(batch)}/{len(valid_files)}")

print("Indexing complete")
