from scripts.network_simulation import (
    create_applications,
    simulate_separate_databases,
    simulate_shared_database,
)


def test_zero_percent_sharing():
    attacks = [f"attack_{i}" for i in range(20)]

    applications, shared_count = create_applications(attacks, 0)

    assert shared_count == 0

    separate_calls = simulate_separate_databases(applications)
    shared_calls, stored = simulate_shared_database(applications)

    assert separate_calls == 20
    assert shared_calls == 20
    assert stored == 20


def test_fifty_percent_sharing():
    attacks = [f"attack_{i}" for i in range(20)]

    applications, shared_count = create_applications(attacks, 50)

    separate_calls = simulate_separate_databases(applications)
    shared_calls, stored = simulate_shared_database(applications)

    assert shared_count == 10
    assert separate_calls == 40
    assert shared_calls == 20
    assert stored == 20


def test_one_hundred_percent_sharing():
    attacks = [f"attack_{i}" for i in range(20)]

    applications, shared_count = create_applications(attacks, 100)

    separate_calls = simulate_separate_databases(applications)
    shared_calls, stored = simulate_shared_database(applications)

    assert shared_count == 20
    assert separate_calls == 60
    assert shared_calls == 20
    assert stored == 20


def test_every_attack_is_assigned():
    attacks = [f"attack_{i}" for i in range(20)]

    applications, _ = create_applications(attacks, 50)

    all_assigned_attacks = set()

    for app_attacks in applications.values():
        all_assigned_attacks.update(app_attacks)

    assert all_assigned_attacks == set(attacks)


def test_shared_database_avoids_calls():
    attacks = [f"attack_{i}" for i in range(20)]

    applications, _ = create_applications(attacks, 75)

    separate_calls = simulate_separate_databases(applications)
    shared_calls, _ = simulate_shared_database(applications)

    assert shared_calls <= separate_calls
