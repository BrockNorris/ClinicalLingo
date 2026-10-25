"""Entry point: python cl.py <command> ...

Exists so the portable (embeddable) Python can run the package without
installing it: running a script puts this folder on sys.path, whereas
`python -m clinicallingo` would not work from the embeddable build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinicallingo.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
