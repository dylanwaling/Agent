import lancedb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------

uri = "data/lancedb"
db = lancedb.connect(uri)

# Initialize the embedding model (same as used for creating embeddings)
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table("docling")

# --------------------------------------------------------------
# Search the table
# --------------------------------------------------------------

query = "what's docling?"
query_embedding = model.encode(query)

result = table.search(query_embedding).limit(3)
print(result.to_pandas())
