"""Initialize and migrate PostgreSQL tables using SQLAlchemy metadata."""

import sys
from pathlib import Path
from sqlalchemy import inspect, text

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import init_db_tables_sync, sync_engine


def main() -> int:
    print("=" * 60)
    print(" INITIALIZING & MIGRATING POSTGRESQL TABLES")
    print("=" * 60)

    print("[*] Creating database tables if not existing...")
    init_db_tables_sync()

    # Ensure schema migrations for existing tables
    print("[*] Verifying and applying schema column synchronization...")
    with sync_engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS category VARCHAR(50) DEFAULT 'UNKNOWN';
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_engine_used VARCHAR(50);
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata_json JSON;
        """))
        conn.commit()

    inspector = inspect(sync_engine)
    tables = inspector.get_table_names()
    print(f"[*] Verified PostgreSQL tables in database:")
    for t in tables:
        print(f"    - {t}")

    print("\nPostgreSQL schema initialization completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
