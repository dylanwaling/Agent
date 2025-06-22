from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from utils.tokenizer import OpenAITokenizerWrapper


tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length


# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

converter = DocumentConverter()
print("📄 Extracting document for chunking...")
result = converter.convert("https://arxiv.org/pdf/2408.09869")
print("✅ Document extraction completed")


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

print("🔪 Starting document chunking...")
chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

print(f"✅ Chunking completed! Created {len(chunks)} chunks")
print(f"📊 Average chunk size: {sum(len(chunk.text) for chunk in chunks) // len(chunks)} characters")

# Show preview of first chunk
if chunks:
    print(f"\n📝 First chunk preview:")
    print(f"Text: {chunks[0].text[:200]}...")
    print(f"Length: {len(chunks[0].text)} characters")
