# Python Programming Project

Starter structure for an LLM-based application where the **language model tier** and the **guard model** are not decided yet. Both sit behind small interfaces, so they can be chosen later by editing `config/settings.yaml` without touching the rest of the code.

## How it fits together

```
user input -> guard (input check) -> language model -> guard (output check) -> response
                                  \                                        /
                                   +------------- SQLite database --------+
                                         (conversations, messages, guard logs)
```

## Layout

```
python-programming/
├── config/settings.yaml        # model tier, guard model, database path (all placeholders)
├── data/                       # local database lives here (git-ignored)
├── docs/architecture.md        # design notes and open decisions
├── scripts/init_db.py          # creates the database from schema.sql
├── src/app/
│   ├── config.py               # loads settings.yaml + environment variables
│   ├── pipeline.py             # guard -> model -> guard flow
│   ├── main.py                 # command line entry point
│   ├── models/                 # language model interface + registry + echo placeholder
│   ├── guard/                  # guard model interface + registry + allow-all placeholder
│   └── db/                     # schema.sql + database access code
└── tests/                      # pytest tests
```

## Quick start

```bash
cd python-programming
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/init_db.py          # create the database
python -m app.main "Hello there"   # runs with the placeholder model + guard
pytest                             # run the tests
```

## Decisions still open

| Decision | Where to change it | Current placeholder |
|---|---|---|
| Language model tier | `model.provider` / `model.name` in `config/settings.yaml` | `echo` (returns the input) |
| Guard model | `guard.provider` / `guard.name` in `config/settings.yaml` | `allow_all` (passes everything) |
| Database engine | `database.path` / `src/app/db/` | SQLite (standard library, no install) |

To add a real model, create a class in `src/app/models/` that subclasses `LanguageModel`, then register it in `src/app/models/__init__.py`. The guard works the same way.

## Secrets

Copy `.env.example` to `.env` and put API keys there. `.env` is git-ignored, so keys never reach GitHub.
