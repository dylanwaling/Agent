from typing import List, Optional

import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.tokenizer import OpenAITokenizerWrapper

load_dotenv()


tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length


# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------

# Create a LanceDB database
db = lancedb.connect("data/lancedb")

# Initialize the embedding model
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------

# Create embeddings and prepare data for LanceDB
processed_chunks = []
for chunk in chunks:
    # Create embedding for the chunk text
    embedding = model.encode(chunk.text)
    
    # Process metadata
    page_numbers = [
        page_no
        for page_no in sorted(
            set(
                prov.page_no
                for item in chunk.meta.doc_items
                for prov in item.prov
            )
        )
    ] or None
    
    processed_chunks.append({
        "text": chunk.text,
        "vector": embedding,
        "filename": chunk.meta.origin.filename,
        "page_numbers": page_numbers,
        "title": chunk.meta.headings[0] if chunk.meta.headings else None,
    })

# --------------------------------------------------------------
# Add the chunks to the table
# --------------------------------------------------------------

# Create table and add data
table = db.create_table("docling", data=processed_chunks, mode="overwrite")

# --------------------------------------------------------------
# Verify the table
# --------------------------------------------------------------

print(f"Table created with {len(table)} rows")
print(table.to_pandas().head())
