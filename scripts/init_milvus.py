"""Initialize Milvus standalone vector database collection and index."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.indexing.milvus_store import get_milvus_store


def main() -> int:
    print("=" * 60)
    print(" INITIALIZING MILVUS VECTOR COLLECTION & INDEX")
    print("=" * 60)

    store = get_milvus_store()
    print("[*] Connecting to Milvus and configuring collection schema...")
    collection = store.ensure_collection()
    print(f"    --> Collection: '{collection.name}'")
    print(f"    --> Primary Key: {collection.primary_field.name}")
    print(f"    --> Total Entities: {collection.num_entities}")
    print("\nMilvus vector collection successfully initialized and loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
