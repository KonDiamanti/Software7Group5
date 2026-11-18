"""Reusable functions for the English-only rule-based linguistic tier."""

from pathlib import Path
import ast
import re
import time

import pandas as pd


CLASSIFIERS_FILE = Path(__file__).with_name("classifiers.py")
ENGLISH_DATASETS = ("jackhhao", "hackaprompt", "ours")

HIGH_SPECIFICITY_RULES = (
    "no limits",
    "ignore instructions",
    "jailbreak names",
)


def load_rule_definitions():
    """Read RULES from classifiers.py without importing ML dependencies."""
    tree = ast.parse(CLASSIFIERS_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RULES":
                    return ast.literal_eval(node.value)
    raise RuntimeError("Could not find RULES in classifiers.py")


RULES = load_rule_definitions()
COMPILED_RULES = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in RULES.items()
}


def rule_check(text):
    return int(any(rule.search(text) for rule in COMPILED_RULES.values()))


def prepare_english_linguistic_data(data):
    """Keep datasets that are in the project's English-only scope."""
    excluded = sorted(set(data["dataset"].dropna()) - set(ENGLISH_DATASETS))
    english = data[data["dataset"].isin(ENGLISH_DATASETS)].copy()
    if excluded:
        print(
            "Excluded from linguistic scope:",
            ", ".join(excluded),
            "(mixed-language or outside the defined English scope)",
        )
    print(f"English-scope prompts: {len(english)}")
    return english


def load_existing_classifier_prompts(
    predictions_file="results/classifier_predictions.csv",
):
    """Load prompts without downloading datasets or importing the ML stack."""
    predictions_file = Path(predictions_file)
    if not predictions_file.is_absolute():
        predictions_file = Path(__file__).parent / predictions_file

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Cannot find {predictions_file}. Run the existing classifier notebook "
            "once, or provide a predictions CSV."
        )

    predictions = pd.read_csv(predictions_file)
    missing = {"dataset", "text", "label"} - set(predictions.columns)
    if missing:
        raise ValueError(
            f"{predictions_file.name} is missing columns: {', '.join(sorted(missing))}"
        )

    prompts = predictions[["dataset", "text", "label"]].drop_duplicates(
        subset=["dataset", "text"]
    )
    return prepare_english_linguistic_data(prompts)


def matched_rules(text):
    """Return a list of rule names that fire on text."""
    return [name for name, rule in COMPILED_RULES.items() if rule.search(text)]


def rule_feature_matrix(texts):
    """Return one 0/1 column per linguistic rule."""
    rows = []
    for text in texts:
        matches = set(matched_rules(text))
        rows.append({name: int(name in matches) for name in RULES})
    return pd.DataFrame(rows)


def linguistic_rule_check(text, excluded_rules=(), only_rules=None):
    """Apply the selected rules to one prompt."""
    excluded = set(excluded_rules)
    selected = set(only_rules) if only_rules is not None else set(COMPILED_RULES)
    return int(any(
        name not in excluded
        and name in selected
        and rule.search(text)
        for name, rule in COMPILED_RULES.items()
    ))


def high_specificity_check(text):
    """Block only rules with high specificity in the project test data."""
    return linguistic_rule_check(text, only_rules=HIGH_SPECIFICITY_RULES)


def role_play_aware_check(text):
    """Block matched rules except role play used by itself."""
    matches = matched_rules(text)
    return int(any(name != "role play" for name in matches))


def linguistic_check_counts(data):
    """How often each rule fires on attacks and harmless prompts."""
    data = prepare_english_linguistic_data(data)
    jackhhao = data[data["dataset"] == "jackhhao"]
    hackaprompt = data[data["dataset"] == "hackaprompt"]

    def fires(rule, prompts):
        return prompts["text"].apply(lambda text: bool(rule.search(text)))

    return pd.DataFrame([
        {
            "rule": name,
            "attacks (jackhhao)": int(fires(rule, jackhhao)[jackhhao["label"] == 1].sum()),
            "harmless (jackhhao)": int(fires(rule, jackhhao)[jackhhao["label"] == 0].sum()),
            "HackAPrompt attacks": int(fires(rule, hackaprompt[hackaprompt["label"] == 1]).sum()),
            "HackAPrompt harmless": int(fires(rule, hackaprompt[hackaprompt["label"] == 0]).sum()),
        }
        for name, rule in COMPILED_RULES.items()
    ]).sort_values("attacks (jackhhao)", ascending=False)


def run_linguistic_benchmark(data, output_folder="results/linguistic-features",
                             include_rule_ablations=True):
    """Benchmark the original linguistic policies and save CSV outputs."""
    data = prepare_english_linguistic_data(data)
    checks = [
        ("Linguistic rules (baseline)", rule_check),
        ("High-specificity rules", high_specificity_check),
        ("Role-play-aware rules", role_play_aware_check),
    ]

    if include_rule_ablations:
        for rule_name in COMPILED_RULES:
            checks.append((
                f"Rules without {rule_name}",
                lambda text, excluded=rule_name: linguistic_rule_check(
                    text, excluded_rules=(excluded,)
                ),
            ))

    predictions = data.copy()
    result_rows = []
    for name, check in checks:
        answers = []
        scores = []
        times = []
        for text in predictions["text"]:
            start = time.perf_counter()
            result = check(text)
            elapsed = (time.perf_counter() - start) * 1000
            answer, score = result if isinstance(result, tuple) else (result, None)
            answers.append(int(answer))
            scores.append(score)
            times.append(elapsed)

        predictions[name] = answers
        predictions[name + " score"] = [
            float(answer) if score is None else float(score)
            for answer, score in zip(answers, scores)
        ]
        predictions[name + " ms"] = times

        groups = list(predictions.groupby("dataset", sort=False))
        combined = predictions[predictions["dataset"].isin(ENGLISH_DATASETS)]
        groups.append(("all", combined))
        for dataset, part in groups:
            attacks = part["label"] == 1
            flagged = part[name] == 1
            caught = int((attacks & flagged).sum())
            missed = int((attacks & ~flagged).sum())
            false_alarms = int((~attacks & flagged).sum())
            attack_count = int(attacks.sum())
            harmless_count = int((~attacks).sum())
            result_rows.append({
                "classifier": name,
                "dataset": dataset,
                "prompts": len(part),
                "attacks caught": caught,
                "attacks missed": missed,
                "false alarms": false_alarms,
                "detection rate": caught / attack_count if attack_count else None,
                "false alarm rate": false_alarms / harmless_count if harmless_count else None,
                "F1": (
                    2 * caught / (2 * caught + false_alarms + missed)
                    if 2 * caught + false_alarms + missed
                    else None
                ),
                "AUC": None,
                "detection at 1% false alarms": None,
                "ms per prompt": float(part[name + " ms"].mean()),
                "note": "",
                "training s": None,
            })

    bench = LinguisticBenchmark(predictions, pd.DataFrame(result_rows))
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    bench.table().to_csv(output_folder / "classifier_results.csv", index=False)
    bench.predictions.to_csv(output_folder / "classifier_predictions.csv", index=False)
    print(
        "Saved",
        output_folder / "classifier_results.csv",
        "and",
        output_folder / "classifier_predictions.csv",
    )
    return bench


class LinguisticBenchmark:
    def __init__(self, predictions, results):
        self.predictions = predictions
        self.results = results

    def table(self):
        return self.results


def print_role_play_analysis(bench):
    """Print false-alarm counts for role-play prompts before and after routing."""
    data = bench.predictions
    for name in (
        "Linguistic rules (baseline)",
        "Role-play-aware rules",
    ):
        false_alarms = data[(data["label"] == 0) & (data[name] == 1)]
        role_play_false_alarms = sum(
            "role play" in matched_rules(text)
            for text in false_alarms["text"]
        )
        print(
            f"{name}: {len(false_alarms)} false alarms; "
            f"{role_play_false_alarms} match the role-play rule"
        )


if __name__ == "__main__":
    existing_predictions = Path(__file__).with_name("results") / "classifier_predictions.csv"
    if existing_predictions.exists():
        existing = pd.read_csv(existing_predictions)
        test_data = existing[["dataset", "text", "label"]].drop_duplicates(
            subset=["dataset", "text"]
        )
    else:
        from prompt_data import load_test_prompts
        test_data = load_test_prompts(hackaprompt_sample=500)
    print(test_data.groupby(["dataset", "label"], sort=False).size())
    print("\nRule counts:")
    print(linguistic_check_counts(test_data).to_string(index=False))
    benchmark = run_linguistic_benchmark(test_data)
    print("\nRole-play analysis:")
    print_role_play_analysis(benchmark)
