"""
query_rag.py
-------------
RAG agent using OpenAI to analyze MD&A sections and generate insights.
"""

import argparse
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv  # for loading .env file

# Load environment variables from .env file
# Try loading from current directory first, then from parent if needed
env_loaded = load_dotenv()
if not env_loaded:
    # Try loading from project root (one level up)
    import pathlib
    project_root = pathlib.Path(__file__).parent
    env_loaded = load_dotenv(dotenv_path=project_root / ".env")

# -----------------------------
# CONFIG
# -----------------------------
INDEX_NAME = "mda-index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo", "gpt-4", etc.

# -----------------------------
# Initialize Pinecone (using new API)
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -----------------------------
# Initialize embedding model
# -----------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------------
# Initialize OpenAI client
# -----------------------------
# Get and clean the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip()  # Remove any leading/trailing whitespace

# Only pass api_key if we have a valid (non-empty) key
# Otherwise, let OpenAI read from environment variable automatically
if OPENAI_API_KEY and len(OPENAI_API_KEY) > 0:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    # Try letting OpenAI read from environment automatically
    # This handles cases where the key might be set as a system environment variable
    try:
        openai_client = OpenAI()  # Will read from OPENAI_API_KEY env var automatically
    except Exception as e:
        # If that fails, provide helpful error
        import pathlib
        env_path = pathlib.Path(__file__).parent / ".env"
        raise ValueError(
            f"OPENAI_API_KEY not found or invalid.\n\n"
            f"Please ensure:\n"
            f"1. Your .env file exists at: {env_path}\n"
            f"2. It contains: OPENAI_API_KEY=sk-... (no spaces around =)\n"
            f"3. The key is valid and not empty\n\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Script location: {pathlib.Path(__file__).parent}\n\n"
            f"Original error: {str(e)}"
        ) from e

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve(query, top_k=5):
    """
    Retrieve top_k relevant MD&A chunks from Pinecone.
    """
    query_vec = embed_model.encode(query).tolist()
    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )
    chunks = [match['metadata']['text'] for match in res['matches']]
    return chunks

# -----------------------------
# RAG agent function
# -----------------------------
def generate_insights(query, top_k=5, max_tokens=500):
    """
    Retrieve relevant chunks and generate structured insights using OpenAI.
    """
    retrieved_texts = retrieve(query, top_k=top_k)
    context = "\n\n".join(retrieved_texts)
    
    prompt = f"""Analyze the following MD&A (Management Discussion & Analysis) sections and answer the query.

Query: {query}

Relevant MD&A sections:
{context}

Please provide a structured and insightful answer based on the retrieved information."""
    
    # Call OpenAI
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a financial analyst expert at analyzing SEC 10-K filings and MD&A sections."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# Command-line interface
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, required=True, help="Query string")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens for OpenAI output")
    args = parser.parse_args()

    output = generate_insights(args.query, top_k=args.top_k, max_tokens=args.max_tokens)
    print("\n Generated Insights \n")
    print(output)

if __name__ == "__main__":
    main()
