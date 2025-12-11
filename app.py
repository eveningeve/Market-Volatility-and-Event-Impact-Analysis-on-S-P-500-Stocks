"""
streamlit_app.py
----------------
Streamlit interface for MD&A RAG agent
"""

import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
INDEX_NAME = "mda-index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="MD&A RAG Agent", page_icon="📊", layout="wide")

# -----------------------------
# Initialize clients (with caching)
# -----------------------------
@st.cache_resource
def init_clients():
    """Initialize Pinecone, embedding model, and OpenAI client"""
    # Get API keys from Streamlit secrets (cloud) or .env (local)
    try:
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        openai_key = st.secrets["OPENAI_API_KEY"]
    except:
        # Fallback to .env for local development
        from dotenv import load_dotenv
        load_dotenv()
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not pinecone_key or not openai_key:
            raise ValueError("API keys not found in secrets or .env")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)
    
    # Initialize embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    # Initialize OpenAI
    openai_client = OpenAI(api_key=openai_key)
    
    return index, embed_model, openai_client

# Load clients
try:
    index, embed_model, openai_client = init_clients()
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

# -----------------------------
# Core functions
# -----------------------------
def retrieve(query, top_k=5):
    """Retrieve top_k relevant MD&A chunks from Pinecone"""
    query_vec = embed_model.encode(query).tolist()
    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )
    chunks = [match['metadata']['text'] for match in res['matches']]
    return chunks

def generate_insights(query, top_k=5, max_tokens=500):
    """Retrieve relevant chunks and generate insights using OpenAI"""
    retrieved_texts = retrieve(query, top_k=top_k)
    context = "\n\n".join(retrieved_texts)
    
    prompt = f"""Analyze the following MD&A (Management Discussion & Analysis) sections and answer the query.

Query: {query}

Relevant MD&A sections:
{context}

Please provide a structured and insightful answer based on the retrieved information."""
    
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a financial analyst expert at analyzing SEC 10-K filings and MD&A sections."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip(), retrieved_texts

# -----------------------------
# UI
# -----------------------------
st.title("MD&A RAG Agent")
st.markdown("Ask questions about Management Discussion & Analysis sections from SEC filings")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
    max_tokens = st.slider("Max response tokens", 100, 1000, 500, step=50)
    show_sources = st.checkbox("Show retrieved sources", value=False)

# Main input
query = st.text_input("Enter your query:", placeholder="e.g., What are the main risks mentioned?")

if st.button("Analyze", type="primary"):
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Analyzing..."):
            try:
                insights, sources = generate_insights(query, top_k=top_k, max_tokens=max_tokens)
                
                st.subheader("Insights")
                st.markdown(insights)
                
                if show_sources:
                    st.subheader("Retrieved Sources")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i}"):
                            st.text(source)
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")