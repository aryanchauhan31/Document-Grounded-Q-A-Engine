"""
One-off batch job: load the real CUAD legal-contract corpus, chunk it, embed
it, and upsert into Pinecone. Also writes a sampled QA eval set to disk
(questions/answers pairs grounded in the real contracts) for the faithfulness/
relevancy eval, mirroring engine.py's questions.json/answers.json pattern.
"""
import json
import os

from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec

from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

INDEX_NAME = "cuad-legal-rag"
EMBED_DIM = 384  # BAAI/bge-small-en-v1.5

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

print("Loading chenghao/cuad_qa ...")
ds = load_dataset("chenghao/cuad_qa", split="train")

# QA rows share the same context per contract (CUAD asks ~41 clause
# categories per contract) -- dedupe down to one Document per real contract.
contracts = {}
for row in ds:
    contracts.setdefault(row["title"], row["context"])
print(f"Unique contracts: {len(contracts)}")

documents = [
    Document(text=text, metadata={"title": title})
    for title, text in contracts.items()
]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    print(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM}) ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
pinecone_index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Chunking + embedding + upserting into Pinecone (this will take a while) ...")
VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)
print("Ingestion complete.")
print(pinecone_index.describe_index_stats())

# Sample a QA eval set grounded in the real contracts, for faithfulness/
# relevancy scoring later -- same shape as engine.py's questions.json.
eval_rows = []
seen_titles = set()
for row in ds:
    if row["title"] in seen_titles:
        continue
    if not row["answers"]["text"]:
        continue
    seen_titles.add(row["title"])
    eval_rows.append({
        "question": row["question"] + f" (in the contract titled: {row['title']})",
        "expected_answer": row["answers"]["text"][0],
        "title": row["title"],
    })
    if len(eval_rows) >= 50:
        break

with open("/workspace/cuad_eval_questions.json", "w") as f:
    json.dump(eval_rows, f, indent=2)
print(f"Wrote {len(eval_rows)} eval questions to /workspace/cuad_eval_questions.json")
