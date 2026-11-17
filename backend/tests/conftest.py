import os
import sys

# Permite `import detector` / `import ingest` desde backend/tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
