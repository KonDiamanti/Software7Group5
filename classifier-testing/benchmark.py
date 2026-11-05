"""Runs the classifiers on the test prompts and builds the result tables."""
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from classifiers import COMPILED_RULES

COMBINED = ["deepset", "jackhhao"]   # the "all" rows: the two datasets with attacks and harmless prompts


@contextmanager
def skip_on_error(what):
    """Prints why a classifier was skipped (for example a missing Hugging Face login) and goes on."""
    try:
        yield
    except Exception as error:
        print(f"Skipped {what}:", repr(error))


class Benchmark:
    def __init__(self, data):
        self.data = data
        self.predictions = data.copy()   # one answer, score and time column per classifier
        self.results = {}                # classifier -> one summary row per dataset
        self.trained_on = {}             # classifier -> test datasets it was trained on (not a fair test)
        self.training_times = {}

    def run(self, name, check, trained_on=(), training_time=None):
        check("Hello, how are you?")   # warm-up, so that loading the model is not timed
        if hasattr(check, "odd_answers"):
            check.odd_answers.clear()
        answers, scores, times = [], [], []
        for text in tqdm(self.data["text"], desc=name):
            start = time.perf_counter()
            result = check(text)
            times.append((time.perf_counter() - start) * 1000)
            answer, score = result if isinstance(result, tuple) else (result, None)
            answers.append(int(answer))
            scores.append(score)

        self.predictions[name] = answers
        if any(score is not None for score in scores):
            if None in scores:
                print(f"{scores.count(None)} prompts got no score, so their answer is used as their score.")
            self.predictions[name + " score"] = [float(a) if s is None else s for a, s in zip(answers, scores)]
        self.predictions[name + " ms"] = times
        odd = getattr(check, "odd_answers", [])
        if odd:
            print(f"{len(odd)} answers were not in the expected format, for example {[answer[:100] for answer in odd[:3]]}")

        self.trained_on[name] = list(trained_on)
        if training_time is not None:
            self.training_times[name] = training_time
        parts = list(self.predictions.groupby("dataset", sort=False))
        parts.append(("all", self.predictions[self.predictions["dataset"].isin(COMBINED)]))
        self.results[name] = [self.summarise(name, dataset, part) for dataset, part in parts]
        display(pd.DataFrame(self.results[name]).set_index("dataset").drop(columns="classifier").round(3))

    def summarise(self, name, dataset, part):
        is_attack, flagged = part["label"] == 1, part[name] == 1
        caught = int((is_attack & flagged).sum())
        missed = int((is_attack & ~flagged).sum())
        false_alarms = int((~is_attack & flagged).sum())
        attacks, harmless = int(is_attack.sum()), int((~is_attack).sum())
        row = {
            "classifier": name,
            "dataset": dataset,
            "prompts": len(part),
            "attacks caught": caught,
            "attacks missed": missed,
            "false alarms": false_alarms,
            "detection rate": caught / attacks if attacks else float("nan"),
            "false alarm rate": false_alarms / harmless if harmless else float("nan"),
            "F1": 2 * caught / (2 * caught + false_alarms + missed) if caught + false_alarms + missed else float("nan"),
            "AUC": float("nan"),
            "detection at 1% false alarms": float("nan"),
            "ms per prompt": part[name + " ms"].mean(),
            "note": self.note(name, dataset),
        }
        score_column = name + " score"
        if score_column in part and attacks and harmless:
            row["AUC"] = roc_auc_score(part["label"], part[score_column])
            false_alarm_rates, detection_rates, _ = roc_curve(part["label"], part[score_column], drop_intermediate=False)
            row["detection at 1% false alarms"] = detection_rates[false_alarm_rates <= 0.01].max()
        return row

    def note(self, name, dataset):
        trained = self.trained_on.get(name, [])
        if dataset in trained:
            return "trained on this data"
        if dataset == "all" and set(trained) & set(COMBINED):
            return f"includes data it was trained on ({', '.join(trained)})"
        return ""

    def table(self):
        table = pd.DataFrame([row for rows in self.results.values() for row in rows])
        table["training s"] = table["classifier"].map(self.training_times)
        return table

    def detection_by_dataset(self):
        """Detection rate in % on jailbreaks (jackhhao) and the two sets of prompt injections."""
        detection = 100 * self.table().pivot(index="classifier", columns="dataset", values="detection rate")
        detection = detection.reindex(index=list(self.results), columns=["jackhhao", "deepset", "hackaprompt"])
        return detection.round(1).rename(columns={"jackhhao": "jailbreaks (jackhhao)", "deepset": "injections (deepset)",
                                                  "hackaprompt": "injections (HackAPrompt)"})

    def missed_by_all(self):
        """Do the classifiers miss the same attacks? This matters for one tier against two tiers."""
        names = list(self.results)
        attacks = self.predictions[self.predictions["dataset"].isin(COMBINED) & (self.predictions["label"] == 1)]
        missed = attacks[attacks[names].max(axis=1) == 0]
        print(f"{len(missed)} of the {len(attacks)} attacks in deepset and jackhhao are missed by all {len(names)} classifiers,",
              missed["dataset"].value_counts().to_dict())
        others = [name for name in names if name != "Rule-based check"]
        if "Rule-based check" in names and others:
            rules_missed = attacks[attacks["Rule-based check"] == 0]
            print(int((rules_missed[others].max(axis=1) == 1).sum()), "attacks that the rules miss are caught by another classifier.")
            print(int(((attacks["Rule-based check"] == 1) & (attacks[others].max(axis=1) == 0)).sum()),
                  "attacks are caught only by the rules.")

    def mistakes(self, name, dataset=None, show=10):
        part = self.predictions if dataset is None else self.predictions[self.predictions["dataset"] == dataset]
        missed = part[(part["label"] == 1) & (part[name] == 0)]
        false_alarms = part[(part["label"] == 0) & (part[name] == 1)]
        for title, rows in [("missed attacks", missed), ("false alarms", false_alarms)]:
            print(f"{name}: {len(rows)} {title}" + (f", the first {show}:" if len(rows) > show else ":"))
            for _, row in rows.head(show).iterrows():
                print(f"- [{row['dataset']}] {row['text'][:300]!r}")
            print()

    def overleaf_rows(self, report_names=None):
        """Rows for the report tables. $^{*}$ marks results on data the model was trained on."""
        report_names = report_names or {}
        tables = {
            "tab:results": [("deepset", "detection rate"), ("deepset", "false alarm rate"),
                            ("jackhhao", "detection rate"), ("jackhhao", "false alarm rate"),
                            ("all", "F1"), ("all", "ms per prompt")],
            "tab:scores": [("deepset", "AUC"), ("jackhhao", "AUC"),
                           ("jackhhao", "detection at 1% false alarms"), ("all", "detection at 1% false alarms")],
            "tab:injections": [("deepset", "detection rate"), ("hackaprompt", "detection rate")],
        }
        by_pair = self.table().set_index(["classifier", "dataset"])

        def number(name, dataset, column):
            if (name, dataset) not in by_pair.index or pd.isna(by_pair.loc[(name, dataset), column]):
                return "--"
            value = by_pair.loc[(name, dataset), column]
            if column == "ms per prompt":
                return f"{value:.1f}"
            text = f"{value:.2f}" if column in ("F1", "AUC") else f"{100 * value:.1f}"
            return text + ("$^{*}$" if self.note(name, dataset) else "")

        for label, columns in tables.items():
            print(label + ":", " & ".join(f"{column} ({dataset})" for dataset, column in columns))
            for name in self.results:
                if label == "tab:scores" and pd.isna(by_pair.loc[(name, "all"), "AUC"]):
                    continue   # no score, like the rules
                numbers = [number(name, dataset, column) for dataset, column in columns]
                print(" & ".join([report_names.get(name, name), *numbers]) + r" \\")
            print()

    def save(self, folder="results"):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        self.table().to_csv(folder / "classifier_results.csv", index=False)
        self.predictions.to_csv(folder / "classifier_predictions.csv", index=False)
        print("Saved", folder / "classifier_results.csv", "and", folder / "classifier_predictions.csv")


def rule_counts(data):
    """How often each rule fires on attacks and harmless prompts (input for sub-question 1)."""
    combined = data[data["dataset"].isin(COMBINED)]
    hackaprompt = data[data["dataset"] == "hackaprompt"]

    def fires(rule, prompts):
        return prompts["text"].apply(lambda text: bool(rule.search(text)))

    counts = pd.DataFrame([
        {
            "rule": name,
            "attacks (deepset + jackhhao)": int(fires(rule, combined)[combined["label"] == 1].sum()),
            "harmless (deepset + jackhhao)": int(fires(rule, combined)[combined["label"] == 0].sum()),
            "HackAPrompt attacks": int(fires(rule, hackaprompt).sum()),
        }
        for name, rule in COMPILED_RULES.items()
    ])
    return counts.sort_values("attacks (deepset + jackhhao)", ascending=False)
