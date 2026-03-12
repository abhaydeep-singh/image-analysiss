
import torch
import open_clip
from PIL import Image
import os
import chromadb

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)


model = model.to(device)

# create vector DB
# client = chromadb.Client()
# collection = client.create_collection("images")

# create persistance
client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_or_create_collection(
    "images",
    metadata={"hnsw:space": "cosine"} # use cosine
)

folder = "./assets"

for file in os.listdir(folder):
    if file.lower().endswith((".png",".jpg",".jpeg")):

        path = os.path.join(folder,file)

        image = preprocess(Image.open(path)).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        collection.add(
            ids=[file],
            embeddings=[embedding.cpu().numpy()[0]],
            metadatas=[{"path": path}]
        )

print("Images indexed")