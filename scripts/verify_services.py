"""Phase 1 Environment & Services Verification Script.

Tests Python version, core imports, Tesseract binary, and Docker service ports.
"""

import os
import sys
import socket
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version() -> bool:
    print("\n[1/5] Checking Python Version...")
    v = sys.version_info
    print(f"      Detected: Python {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and v.minor >= 12:
        print("      --> SUCCESS: Python 3.12+ requirement satisfied.")
        return True
    else:
        print("      --> ERROR: Python 3.12 or higher is required.")
        return False


def check_core_modules() -> bool:
    print("\n[2/5] Checking Core Application Modules...")
    try:
        from app.core.constants import TaskStatus, MIME_TYPE_MAPPING, SUPPORTED_EXTENSIONS
        from app.core.config import get_settings
        from app.core.logging import setup_logging, get_logger
        from app.core.exceptions import AppException

        setup_logging("INFO")
        logger = get_logger("verification")
        settings = get_settings()

        print(f"      Loaded App: '{settings.APP_NAME}' (Env: {settings.APP_ENV})")
        print(f"      Supported MIME Types Count: {len(MIME_TYPE_MAPPING)}")
        print(f"      Task Lifecycle States: {[s.value for s in TaskStatus]}")
        print("      --> SUCCESS: Core modules imported and configured cleanly.")
        return True
    except Exception as exc:
        print(f"      --> ERROR importing core modules: {exc}")
        return False


def check_tesseract() -> bool:
    print("\n[3/5] Checking Tesseract OCR Binary...")
    from app.core.config import get_settings
    settings = get_settings()
    tess_path = settings.TESSERACT_CMD

    # Check if configured path exists or is in PATH
    executable = tess_path if os.path.exists(tess_path) else "tesseract"
    try:
        res = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True)
        version_line = res.stdout.strip().split("\n")[0]
        print(f"      Found Tesseract: {version_line} (Path: {executable})")
        print("      --> SUCCESS: Tesseract OCR is available.")
        return True
    except Exception as exc:
        print(f"      --> WARNING: Tesseract not executable via '{executable}': {exc}")
        print("                   Ensure Tesseract is installed and path in .env is correct.")
        return False


def check_port(host: str, port: int, service_name: str) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect((host, port))
        s.close()
        print(f"      [OK]   {service_name:<20} at {host}:{port} is REACHABLE.")
        return True
    except Exception:
        print(f"      [FAIL] {service_name:<20} at {host}:{port} is NOT reachable.")
        return False


def check_infrastructure_services() -> bool:
    print("\n[4/5] Checking Infrastructure Service Ports...")
    from app.core.config import get_settings
    settings = get_settings()

    services = [
        (settings.POSTGRES_HOST, settings.POSTGRES_PORT, "PostgreSQL (5433)"),
        (settings.REDIS_HOST, settings.REDIS_PORT, "Redis (6379)"),
        ("localhost", 9000, "MinIO API (9000)"),
        (settings.MILVUS_HOST, settings.MILVUS_PORT, "Milvus (19530)"),
    ]

    all_passed = True
    for host, port, name in services:
        if not check_port(host, port, name):
            all_passed = False
    return all_passed


def check_folder_structure() -> bool:
    print("\n[5/5] Verifying Folder Structure...")
    required_dirs = [
        "app/api/routes",
        "app/core",
        "app/schemas",
        "app/db/migrations",
        "app/storage",
        "app/queue",
        "app/ingestion",
        "app/ocr/engines",
        "app/processing",
        "app/indexing",
        "app/rag",
        "app/agents",
        "app/services",
        "app/reliability",
        "frontend/components",
        "evaluation/datasets",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "scripts",
        "sample_documents/pdf",
        "sample_documents/images",
        "sample_documents/office",
        "sample_documents/expected_outputs",
        "data",
        "artifacts",
        "docs",
    ]

    missing = []
    for d in required_dirs:
        p = PROJECT_ROOT / d
        if not p.is_dir():
            missing.append(d)

    if not missing:
        print(f"      All {len(required_dirs)} required directories are verified.")
        print("      --> SUCCESS: Workspace structure matches blueprint.")
        return True
    else:
        print(f"      --> ERROR: Missing directories: {missing}")
        return False


def main() -> int:
    print("=" * 70)
    print(" DISTRIBUTED OCR AGENTIC RAG - PHASE 1 VERIFICATION")
    print("=" * 70)

    results = [
        check_python_version(),
        check_core_modules(),
        check_tesseract(),
        check_infrastructure_services(),
        check_folder_structure()
    ]

    print("\n" + "=" * 70)
    if all(results):
        print(" PHASE 1 VERIFICATION: ALL CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print(" PHASE 1 VERIFICATION: COMPLETED WITH WARNINGS/FAILURES.")
        print(" Please review the logs above before proceeding to Phase 2.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
