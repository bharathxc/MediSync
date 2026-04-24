"""
MediSync AI — PDF Ingestion Pipeline
Loads plan PDFs, splits into chunks, and stores in ChromaDB vector database.
"""
import os
import sys
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from config import KNOWLEDGE_BASE_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdfs(directory: str) -> list[dict]:
    """Load all PDFs from directory and extract text with metadata."""
    documents = []

    for filename in os.listdir(directory):
        if not filename.lower().endswith('.pdf'):
            continue

        filepath = os.path.join(directory, filename)
        print(f"  📄 Loading: {filename}")

        try:
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Determine plan type from filename/content
                    plan_type = "Unknown"
                    if "basic" in filename.lower() or "basic" in text.lower()[:500]:
                        plan_type = "Basic"
                    elif "premium" in filename.lower() or "premium" in text.lower()[:500]:
                        plan_type = "Premium"
                    elif "medicare" in filename.lower():
                        plan_type = "Medicare"

                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "plan_type": plan_type,
                        }
                    })
        except Exception as e:
            print(f"  ⚠️  Error loading {filename}: {e}")

    return documents


def split_documents(documents: list[dict]) -> list[dict]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                }
            })

    return chunks


def store_in_chromadb(chunks: list[dict]) -> chromadb.Collection:
    """Store text chunks in ChromaDB with metadata."""
    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection if it exists (fresh ingestion)
    try:
        client.delete_collection("medisync_plans")
        print("  Cleared existing collection")
    except Exception:
        pass

    # Create new collection
    collection = client.create_collection(
        name="medisync_plans",
        metadata={"description": "Optum health plan documents and Medicare forms"}
    )

    # Prepare batch data
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i:04d}")
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    # Add to collection in batches
    batch_size = 50
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


def run_ingestion():
    """Execute the full ingestion pipeline."""
    print("\n" + "=" * 60)
    print("🏥 MediSync AI — Document Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load PDFs
    print("\n📂 Step 1: Loading PDFs from knowledge base...")
    documents = load_pdfs(KNOWLEDGE_BASE_DIR)
    print(f"   Loaded {len(documents)} pages from PDFs")

    if not documents:
        print("❌ No documents found! Place PDF files in the knowledge_base/ directory.")
        sys.exit(1)

    # Step 2: Split into chunks
    print("\n✂️  Step 2: Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"   Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: Store in ChromaDB
    print("\n💾 Step 3: Storing in ChromaDB vector database...")
    collection = store_in_chromadb(chunks)
    print(f"   Stored {collection.count()} chunks in ChromaDB")

    # Step 4: Verification
    print("\n✅ Step 4: Verification...")
    test_results = collection.query(
        query_texts=["insulin coverage copay"],
        n_results=3
    )
    print(f"   Test query 'insulin coverage copay' returned {len(test_results['documents'][0])} results")
    for i, (doc, meta) in enumerate(zip(test_results['documents'][0], test_results['metadatas'][0])):
        print(f"   [{i+1}] Source: {meta['source']} (Page {meta['page']}, {meta['plan_type']} plan)")
        print(f"       Preview: {doc[:100]}...")

    print("\n" + "=" * 60)
    print("✅ Ingestion complete! Vector database ready at: " + CHROMA_DB_DIR)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_ingestion()
