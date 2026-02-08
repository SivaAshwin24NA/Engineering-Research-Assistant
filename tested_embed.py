import os
from dotenv import load_dotenv

# ==============================
# LOAD ENV
# ==============================

load_dotenv()

print("✅ Environment loaded")

# ==============================
# TEST LOCAL EMBEDDINGS
# ==============================

from langchain_community.embeddings import HuggingFaceEmbeddings

print("🧠 Loading FREE local embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Model loaded successfully!")

# Test embedding
vector = embeddings.embed_query("Hello engineering world!")

print("\n🔥 EMBEDDING WORKED!")
print("Vector length:", len(vector))
print("First 5 numbers:", vector[:5])
