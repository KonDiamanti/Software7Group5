
"""Import labelled adversarial prompts into the shared attack database."""

from datasets import load_dataset

from src.attack_db.hash_db import HashAttackDB


DATASET_NAME = "Necent/llm-jailbreak-prompt-injection-dataset"
DATABASE_PATH = "attacks.db"
MAX_IMPORTS = 20

def import_attacks():
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    db = HashAttackDB(DATABASE_PATH)

    imported = 0
    skipped = 0

    for row in dataset:
        # Only import prompts explicitly labelled as adversarial.
        if row["prompt_adversarial"] != 1:
            skipped += 1
            continue

        prompt = row["prompt"]

        if not prompt or not prompt.strip():
            skipped += 1
            continue

        source = row["source"] or DATASET_NAME

        db.add(prompt, source=source)
        imported += 1

        if imported >= MAX_IMPORTS:
         break

    print("Import finished!")
    print("Database entries:", db.count())
    print("Adversarial prompts processed:", imported)
    print("Rows skipped:", skipped)

    db.con.close()


if __name__ == "__main__":
    import_attacks()
