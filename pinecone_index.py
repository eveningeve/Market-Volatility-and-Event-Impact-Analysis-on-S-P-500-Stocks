# build_pinecone_index.py

import os
import glob
import uuid
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./data/mda_extracted"
INDEX_NAME = "mda-index"
CHUNK_SIZE = 700        # approx tokens (word-based approximation)
CHUNK_OVERLAP = 150

# Pinecone Cloud Provider & Region Configuration
# ===============================================
# Cloud Provider Options:
#   - "aws"  : Amazon Web Services (14% more expensive than GCP)
#   - "gcp"  : Google Cloud Platform (cheapest option)
#   - "azure": Microsoft Azure
#
# Important Notes:
#   - Starter plan users: MUST use "aws" and "us-east-1" region
#   - Standard/Enterprise: Can choose any provider/region
#   - Cost: GCP is ~14% cheaper than AWS for serverless
#   - Performance: Use same cloud/region as your app for lower latency
#   - Common AWS regions: us-east-1, us-west-2, eu-west-1
#   - Common GCP regions: us-central1, us-east1, europe-west1
#
PINECONE_CLOUD = "aws"  # Change to "gcp" or "azure" if needed
PINECONE_REGION = "us-east-1"  # Change based on your needs/location

# -----------------------------
# INIT
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if index exists, create if it doesn't
index_names = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in index_names:
    print(f"Index '{INDEX_NAME}' not found. Creating it now...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        metric="cosine",  # cosine similarity is standard for text embeddings
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )
    print(f"✅ Index '{INDEX_NAME}' created successfully!")
else:
    print(f"✅ Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)


# -----------------------------
# Chunking function
# -----------------------------
def chunk_text(text: str, max_tokens=700, overlap=150) -> List[str]:
    words = text.split()
    max_words = max_tokens
    overlap_words = overlap

    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap_words

    return chunks


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_all_mda_files():

    files = glob.glob(os.path.join(DATA_DIR, "**/*.mda"), recursive=True)

    print(f"Found {len(files)} MD&A text files.")

    batch_vectors = []
    batch_metadata = []
    batch_ids = []

    BATCH_SIZE = 60  # ok for Pinecone upserts

    for filepath in tqdm(files, desc="Processing files"):

        rel = os.path.relpath(filepath, DATA_DIR)
        parts = rel.split(os.sep)

        # infer metadata from folder structure
        if len(parts) >= 3 and parts[0].isdigit():
            cik = parts[0]
            year = parts[1]
        else:
            cik = "unknown"
            year = "unknown"

        company = os.path.basename(filepath).split("_")[0]
        filename = os.path.basename(filepath)

        text = open(filepath, "r", encoding="utf-8", errors="ignore").read()
        chunks = chunk_text(text, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            vid = str(uuid.uuid4())
            embedding = model.encode(chunk).tolist()

            metadata = {
                "company": company,
                "cik": cik,
                "year": year,
                "filename": filename,
                "chunk_id": idx,
                "text": chunk
            }

            batch_ids.append(vid)
            batch_vectors.append(embedding)
            batch_metadata.append(metadata)

            # batch upsert
            if len(batch_ids) >= BATCH_SIZE:
                index.upsert(vectors=list(zip(batch_ids, batch_vectors, batch_metadata)))
                batch_ids, batch_vectors, batch_metadata = [], [], []

    # final leftovers
    if batch_ids:
        index.upsert(vectors=list(zip(batch_ids, batch_vectors, batch_metadata)))

    print("\n🎉 Finished indexing all MD&A chunks into Pinecone!")


if __name__ == "__main__":
    process_all_mda_files()
