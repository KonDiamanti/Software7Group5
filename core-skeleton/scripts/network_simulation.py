
import argparse
from datasets import load_dataset
from src.attack_db.hash_db import HashAttackDB

DATASET_NAME = "Necent/llm-jailbreak-prompt-injection-dataset"


def load_real_attacks(limit):
    """Load unique adversarial prompts from the real dataset."""
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    attacks = []
    seen = set()

    for row in dataset:
        if row["prompt_adversarial"] != 1:
            continue

        prompt = row["prompt"]

        if not prompt or not prompt.strip():
            continue

        if prompt in seen:
            continue

        attacks.append(prompt)
        seen.add(prompt)

        if len(attacks) >= limit:
            break

    if len(attacks) < limit:
        raise ValueError(
            f"Only found {len(attacks)} unique attacks, "
            f"but {limit} were requested."
        )

    return attacks


def create_applications(attacks, sharing_percentage):
    """
    A percentage of attacks is shared by all three applications.
    Remaining attacks are assigned to just one application.
    """
    shared_count = round(
        len(attacks) * sharing_percentage / 100
    )

    shared_attacks = attacks[:shared_count]
    private_attacks = attacks[shared_count:]

    applications = {
        "App A": list(shared_attacks),
        "App B": list(shared_attacks),
        "App C": list(shared_attacks),
    }

    # Distribute private attacks across the three applications.
    for index, attack in enumerate(private_attacks):
        app_name = ["App A", "App B", "App C"][index % 3]
        applications[app_name].append(attack)

    return applications, shared_count


def simulate_separate_databases(applications):
    """Each application has its own independent database."""
    guard_calls = 0

    for app_name, app_attacks in applications.items():
        db = HashAttackDB()

        for attack in app_attacks:
            if db.lookup(attack) is None:
                guard_calls += 1
                db.add(attack, source=app_name)

        db.con.close()

    return guard_calls


def simulate_shared_database(applications):
    """All applications use the same database."""
    guard_calls = 0
    db = HashAttackDB()

    for app_name, app_attacks in applications.items():
        for attack in app_attacks:
            if db.lookup(attack) is None:
                guard_calls += 1
                db.add(attack, source=app_name)

    db_count = db.count()
    db.con.close()

    return guard_calls, db_count


def run_experiment(number_of_attacks):
    attacks = load_real_attacks(number_of_attacks)

    print("\nNETWORK EFFECT EXPERIMENT")
    print("Dataset:", DATASET_NAME)
    print("Unique adversarial prompts:", len(attacks))
    print("Applications: 3")

    print("\nResults:")
    print("-" * 85)
    print(
        f"{'Sharing':<12}"
        f"{'Separate calls':<18}"
        f"{'Shared calls':<16}"
        f"{'Calls avoided':<17}"
        f"{'Reduction':<12}"
    )
    print("-" * 85)

    for sharing_percentage in [0, 25, 50, 75, 100]:
        applications, shared_count = create_applications(
            attacks,
            sharing_percentage
        )

        separate_calls = simulate_separate_databases(
            applications
        )

        shared_calls, stored_attacks = simulate_shared_database(
            applications
        )

        calls_avoided = separate_calls - shared_calls

        reduction = (
            calls_avoided / separate_calls * 100
            if separate_calls > 0
            else 0
        )

        print(
            f"{sharing_percentage}%{'':<8}"
            f"{separate_calls:<18}"
            f"{shared_calls:<16}"
            f"{calls_avoided:<17}"
            f"{reduction:.1f}%"
        )

        print(
            f"  Attacks shared by all apps: {shared_count}; "
            f"unique database entries: {stored_attacks}"
        )

    print("-" * 85)
    print("\nExperiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attacks",
        type=int,
        default=500,
        choices=[20, 50, 100, 500]
    )

    args = parser.parse_args()
    run_experiment(args.attacks)
