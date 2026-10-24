"""Create the database and tables:  python scripts/init_db.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.config import load_settings  # noqa: E402
from app.db import Database  # noqa: E402


def main() -> None:
    settings = load_settings()
    db = Database(settings.database_path)
    db.init_schema()
    db.close()
    print(f"Database ready at {settings.database_path}")


if __name__ == "__main__":
    main()
