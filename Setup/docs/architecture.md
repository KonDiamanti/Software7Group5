# Architecture notes

## Goal

Build the structure first, choose the language model tier and guard model later.

## Components

- **LanguageModel** (`src/app/models/base.py`): one method, `generate(prompt) -> str`. Any provider can be wrapped in this.
- **Guard** (`src/app/guard/base.py`): one method, `check(text, direction) -> GuardResult`. Returns allowed or blocked, with a reason and an optional score.
- **Pipeline** (`src/app/pipeline.py`): checks the input, calls the model, checks the output, and stores everything in the database. Blocked input never reaches the model.
- **Database** (`src/app/db/`): SQLite through the standard library. Schema is in `schema.sql`.

## Database tables

| Table | Purpose |
|---|---|
| `conversations` | One row per conversation |
| `messages` | User and assistant messages, linked to a conversation |
| `guard_logs` | Every guard decision (input or output), linked to a message when one exists |

## Open questions

1. Which language model tier (cost vs quality)?
2. Which guard model, and should it run on input only or on both input and output?
3. Is SQLite enough, or will the project need PostgreSQL later (multiple users, hosting)?
4. What should happen when the guard blocks something: refuse silently, show a message, or escalate to a human?
5. Do conversations need user accounts?

## Moving to another database later

All database access goes through `src/app/db/database.py`, so swapping SQLite for PostgreSQL means changing that one module and `schema.sql`.
