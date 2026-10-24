"""Command line entry point:  python -m app.main "your prompt" """

from __future__ import annotations

import argparse

from .config import load_settings
from .db import Database
from .guard import build_guard
from .models import build_model
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one prompt through the pipeline.")
    parser.add_argument("prompt", help="Text to send to the model")
    parser.add_argument("--conversation", type=int, help="Continue an existing conversation id")
    args = parser.parse_args(argv)

    settings = load_settings()
    db = Database(settings.database_path)
    db.init_schema()

    pipeline = Pipeline(
        model=build_model(settings.model),
        guard=build_guard(settings.guard),
        db=db,
        check_input=settings.guard.check_input,
        check_output=settings.guard.check_output,
    )
    result = pipeline.run(args.prompt, args.conversation)
    print(result.reply)
    print(f"(conversation {result.conversation_id}, blocked={result.blocked})")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
