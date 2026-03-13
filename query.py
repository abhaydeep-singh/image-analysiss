import torch
import open_clip
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

THRESHOLD = 0.20

# load model
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

# correct tokenizer
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)

# connect to DB
client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_collection("images")

query = "person with gun"

# tokenize text
text = tokenizer([query]).to(device)

with torch.no_grad():
    text_embedding = model.encode_text(text)

text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

results = collection.query(
    query_embeddings=[text_embedding.cpu().numpy()[0]],
    n_results=5,
    include=["distances","metadatas"]
)


for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
    confidence = 1 - dist
    status = "✅" if confidence >= THRESHOLD else "❌"
    print(f"{status} {confidence:.4f}  {meta['path']}")